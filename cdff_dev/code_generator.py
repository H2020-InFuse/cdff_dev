import warnings
import os
import jinja2
import glob
from .description_files import validate_node, validate_dfpc


# https://en.cppreference.com/w/cpp/language/types
PRIMITIVE_CPP_TYPES = [
    "bool",
    "char",
    "signed char",
    "unsigned char",
    "wchar_t",
    "char8_t",
    "char16_t",
    "char32_t",
    "short",
    "unsigned short",
    "int",
    "unsigned int",
    "long",
    "unsigned long",
    "long long",
    "unsigned long long",
    "float",
    "double",
    "long double",
]
PRIMITIVE_INITIAL_VALUES = {
    "bool": "false",
    "char": "0",
    "signed char": "0",
    "unsigned char": "0u",
    "wchar_t": "0",
    "char8_t": "0",
    "char16_t": "0",
    "char32_t": "0",
    "short": "0",
    "unsigned short": "0u",
    "int": "0",
    "unsigned int": "0u",
    "long": "0l",
    "unsigned long": "0ul",
    "long long": "0ll",
    "unsigned long long": "ull",
    "float": "0.0f",
    "double": "0.0",
    "long double": "0.0l",
}


# TODO refactor: move type info to some other file
class BasicTypeInfo(object):
    BASICTYPES = PRIMITIVE_CPP_TYPES
    """Information about basic C++ types."""
    def __init__(self, typename):
        self.typename = typename

    @classmethod
    def handles(cls, typename, cdffpath, public_interface):
        return typename in cls.BASICTYPES

    def include(self):
        """C++ include header."""
        return None

    def cython_type(self):
        return self.typename

    def python_type(self):
        return self.typename

    def copy_on_assignment(self):
        return True

    def has_cdfftype(self):
        return False

    def generate_preconstructor_initialization(self, variable_name):
        return "%s(%s)" % (variable_name,
                           PRIMITIVE_INITIAL_VALUES[self.typename])

    def generate_inconstructor_initialization(self, variable_name):
        return ""


class DefaultTypeInfo(object):
    """Handles any type."""
    def __init__(self, typename):
        self.typename = typename
        warnings.warn(
            "Typename '%s' is not known. It is probably not handled "
            "correctly. You might have to add an include to the C++ template "
            "and you might have to change the Python wrapper." % self.typename)

    @classmethod
    def handles(cls, typename, cdffpath, public_interface):
        return not public_interface

    def include(self):
        """C++ include header."""
        return None

    def cython_type(self):
        return self.typename

    def python_type(self):
        return self.typename

    def copy_on_assignment(self):
        return True  # TODO really?

    def has_cdfftype(self):
        return False

    def generate_preconstructor_initialization(self, variable_name):
        return "%s()" % variable_name

    def generate_inconstructor_initialization(self, variable_name):
        return ""


class ASN1TypeInfo(object):
    """Information about ASN1 types."""
    ASN1_TYPES = {}

    def __init__(self, typename):
        self.typename = typename

    @classmethod
    def handles(cls, typename, cdffpath, public_interface):
        if typename[:7] == 'asn1Scc':
            return typename[7:] in cls._search_asn1_type(
                cdffpath, public_interface)

    def include(self):
        """C++ include header."""
        for key in self.ASN1_TYPES.keys():
            if self.typename[7:] in self.ASN1_TYPES[key]:
                return key.split('.')[0] + ".h"

    def cython_type(self):
        return "_cdff_types." + self.typename

    def python_type(self):
        return "cdff_types." + self.typename[7:]

    def copy_on_assignment(self):
        return False

    @classmethod
    def _search_asn1_type(cls, cdffpath, public_interface):
        """Search generated ASN1 types."""
        filenames = cls._search_asn1_filenames(cdffpath, public_interface)

        asn1_types = {}
        asn1_list = []
        for filename in filenames:
            with open(filename, "r", encoding="utf8") as f:
                file_read = f.read()
            splitted_file = file_read.replace('\n', ' ').split('::=')[:-1]
            types_in_file = []
            for i,f in enumerate(splitted_file):
                asn1_type = list(filter(lambda a: a != '', f.split(' ')))[-1]
                if asn1_type != "DEFINITIONS" and asn1_type not in asn1_list:
                    asn1_list.append(cls._handle_taste_types(asn1_type))
                    types_in_file.append(asn1_type)
            asn1_types[filename.split('/')[-1]] = types_in_file
        cls.ASN1_TYPES = asn1_types
        return asn1_list

    @classmethod
    def _search_asn1_filenames(cls, cdffpath, public_interface):
        search_patterns = [
            os.path.join(cdffpath, "Common/Types/ASN.1/ESROCOS/*/*.asn")]
        if not public_interface:
            search_patterns.append(
                os.path.join(cdffpath, "Common/Types/ASN.1/InFuse/*.asn"))
        filenames = []
        for pattern in search_patterns:
            filenames.extend(glob.glob(pattern))
        return filenames

    @classmethod
    def _handle_taste_types(cls, asn1_type):
        return asn1_type.replace('-', '_')

    def has_cdfftype(self):
        return True

    def generate_preconstructor_initialization(self, variable_name):
        return ""

    def generate_inconstructor_initialization(self, variable_name):
        return "%s_Initialize(&%s);" % (self.typename, variable_name)


class TypeRegistry(object):
    TYPEINFOS = [BasicTypeInfo, ASN1TypeInfo, DefaultTypeInfo]
    """Registry for InFuse type information.

    Parameters
    ----------
    cdffpath : str
        Path to CDFF source code

    public_interface : bool
        Not all types are allowed at public interfaces, but any type is allowed
        at an internal interface.
    """
    def __init__(self, cdffpath, public_interface):
        self.cache = {}
        self.cdffpath = cdffpath
        self.public_interface = public_interface

    def get_info(self, typename):
        for TypeInfo in self.TYPEINFOS:
            type_found = TypeInfo.handles(
                typename, self.cdffpath, self.public_interface)
            if type_found:
                if typename not in self.cache:
                    self.cache[typename] = TypeInfo(typename)
                return self.cache[typename]
        else:
            raise TypeError("Type '%s' is not allowed." % typename)


# TODO refactor write_dfn / write_dfpc
def write_dfn(node, cdffpath,
              output, source_folder=".", python_folder="python"):
    """Generate code templates for a data fusion node (DFN).

    Parameters
    ----------
    node : dict
        Node configuration loaded from node description file

    cdffpath : str
        Path to CDFF

    output : str
        Path to output directory

    source_folder : str, optional (default: '.')
        Subdirectory of the output directory that will contain the source code
        template

    python_folder : str, optional (default: 'python')
        Subdirectory of the output directory that will contain the Python
        bindings
    """
    node = validate_node(node)

    type_registry = TypeRegistry(cdffpath, public_interface=False)
    src_dir, python_dir = _prepare_output_directory(
        output, source_folder, python_folder)
    interface_files = write_class(
        node, type_registry, "Interface",
        "%sInterface" % node["name"], target_folder=src_dir,
        force_overwrite=True)
    implementation_files = []
    for implementation in node["implementations"]:
        implementation_files.extend(
            write_class(
                node, type_registry, "Node", implementation,
                target_folder=src_dir))
    cython_files = write_cython(
        node, node["implementations"], type_registry, "Node",
        target_folder=python_dir)
    return interface_files + implementation_files + cython_files


def write_dfpc(dfpc, cdffpath, output, source_folder=".",
               python_folder="python"):
    """Generate code templates for a data fusion processing compound (DFPC).

    Parameters
    ----------
    dfpc : dict
        DFPC configuration loaded from DFPC description file

    cdffpath : str
        Path to CDFF

    output : str
        Path to output directory

    source_folder : str, optional (default: '.')
        Subdirectory of the output directory that will contain the source code
        template

    python_folder : str, optional (default: 'python')
        Subdirectory of the output directory that will contain the Python
        bindings
    """
    dfpc = validate_dfpc(dfpc)

    type_registry = TypeRegistry(cdffpath, public_interface=True)
    src_dir, python_dir = _prepare_output_directory(
        output, source_folder, python_folder)
    interface_files = write_class(
        dfpc, type_registry, "DFPCInterface",
        "%sInterface" % dfpc["name"], target_folder=src_dir,
        force_overwrite=True)
    implementation_files = []
    for implementation in dfpc["implementations"]:
        implementation_files.extend(
            write_class(
                dfpc, type_registry, "DFPCImplementation",
                implementation["name"], target_folder=src_dir))
    implementation_names = [impl["name"] for impl in dfpc["implementations"]]
    cython_files = write_cython(
        dfpc, implementation_names, type_registry, "DFPC",
        target_folder=python_dir)
    return interface_files + implementation_files + cython_files


def _prepare_output_directory(output, source_folder, python_folder):
    src_dir = os.path.join(output, source_folder)
    python_dir = os.path.join(output, python_folder)
    if not os.path.exists(src_dir):
        os.makedirs(src_dir)
    if not os.path.exists(python_dir):
        os.makedirs(python_dir)
    return src_dir, python_dir


def write_class(desc, type_registry, template_base, class_name,
                target_folder="src", force_overwrite=False):
    """Write a C++ class.

    Parameters
    ----------
    desc : dict
        Description of the object that should be generated

    type_registry : TypeRegistry
        Registry for type information

    template_base : str
        Base name of the template that should be used, e.g., 'Node'

    class_name : str
        Name of the class

    target_folder : str, optional (default: 'src')
        Folder where the output should be written

    force_overwrite : bool, optional (default: False)
        Overwrite existing file

    Returns
    -------
    written_files : list
        Names of the files that have been written
    """
    result = {}

    declaration_filename = "%s.hpp" % class_name
    definition_filename = "%s.cpp" % class_name

    includes = set()
    member_initializations = MemberInitializations(type_registry)
    for port in desc["input_ports"]:
        includes.add(type_registry.get_info(port["type"]).include())
        member_initializations.register_input(port["type"], port["name"])
    for port in desc["output_ports"]:
        includes.add(type_registry.get_info(port["type"]).include())
        member_initializations.register_output(port["type"], port["name"])
    if "operations" in desc:
        for op in desc["operations"]:
            for inp in op["inputs"]:
                includes.add(type_registry.get_info(inp["type"]).include())
            includes.add(type_registry.get_info(op["output_type"]).include())
            member_initializations.register_operation(
                op["output_type"], op["name"])

    base_declaration = render(
        "%s.hpp" % template_base, desc=desc, includes=includes,
        class_name=class_name)
    target = os.path.join(target_folder, declaration_filename)
    result[target] = base_declaration

    base_definition = render(
        "%s.cpp" % template_base, declaration_filename=declaration_filename,
        desc=desc, class_name=class_name,
        member_initializations=member_initializations)
    target = os.path.join(target_folder, definition_filename)
    result[target] = base_definition

    return write_result(result, force_overwrite)


class MemberInitializations:
    def __init__(self, type_registry):
        self.type_registry = type_registry
        self.preconstructor_initializations_ = list()
        self.inconstructor_initializations_ = list()

    def register_input(self, typename, name):
        variable_name = "in" + name[0].upper() + name[1:]
        self._register_member(typename, variable_name)

    def register_output(self, typename, name):
        variable_name = "out" + name[0].upper() + name[1:]
        self._register_member(typename, variable_name)

    def register_operation(self, typename, operation_name):
        variable_name = operation_name + "Result"
        self._register_member(typename, variable_name)

    def _register_member(self, typename, variable_name):
        type_info = self.type_registry.get_info(typename)
        prector_init = type_info.generate_preconstructor_initialization(
            variable_name)
        if prector_init:
            self.preconstructor_initializations_.append(prector_init)
        else:
            inctor_init = type_info.generate_inconstructor_initialization(
                variable_name)
            self.inconstructor_initializations_.append(inctor_init)


def write_cython(desc, implementations, type_registry, template_base,
                 target_folder="python", namespace_prefix="dfn_ci"):
    """Write Python binding based on Cython.

    Parameters
    ----------
    desc : dict
        Component description

    implementations : list
        Names of implementations

    type_registry : TypeRegistry
        Registry for type information

    template_base : str
        Base name of the template that should be used, e.g., 'Node'

    target_folder : str, optional (default: 'python')
        Folder where the output should be written

    namespace_prefix : str, optional (default: 'dfn_ci')
        Prefix of namespace

    Returns
    -------
    written_files : list
        Names of the files that have been written
    """
    result = {}

    pxd_filename = "%s.pxd" % desc["name"].lower()
    _pxd_filename = "_%s.pxd" % desc["name"].lower()
    pyx_filename = "%s.pyx" % desc["name"].lower()

    import_cdfftypes = False
    for port in desc["input_ports"] + desc["output_ports"]:
        if type_registry.get_info(port["type"]).has_cdfftype():
            import_cdfftypes = True

    pxd_file = render(
        "Declaration.pxd", desc=desc, implementations=implementations,
        namespace_prefix=namespace_prefix)
    target = os.path.join(target_folder, pxd_filename)
    result[target] = pxd_file

    _pxd_file = render(
        "_%s.pxd" % template_base, desc=desc, implementations=implementations,
        type_registry=type_registry, import_cdfftypes=import_cdfftypes)
    target = os.path.join(target_folder, _pxd_filename)
    result[target] = _pxd_file

    input_ports = render(
        "PythonInputPorts", desc=desc, type_registry=type_registry,
        import_cdfftypes=import_cdfftypes)
    output_ports = render(
        "PythonOutputPorts", desc=desc, type_registry=type_registry,
        import_cdfftypes=import_cdfftypes)

    if "operations" in desc:
        operations = render(
            "PythonOperations", desc=desc, type_registry=type_registry,
            import_cdfftypes=import_cdfftypes)
    else:
        operations = ""

    pyx_file = render(
        "%s.pyx" % template_base, desc=desc, implementations=implementations,
        import_cdfftypes=import_cdfftypes, input_ports=input_ports,
        output_ports=output_ports, operations=operations)

    target = os.path.join(target_folder, pyx_filename)
    result[target] = pyx_file

    return write_result(result, True)


def render(template_name, **kwargs):
    """Render Jinja2 template.

    Parameters
    ----------
    template_name : str
        name of the template, files will be searched in
        $PYTHONPATH/cdff_dev/templates/<template_name>.template
    kwargs : keyword arguments
        arguments that will be passed to the template

    Returns
    -------
    rendered_template : str
        template with context filled in
    """
    env = jinja2.Environment(
        loader=jinja2.PackageLoader("cdff_dev", "templates"),
        lstrip_blocks=True,
        trim_blocks=True,
        keep_trailing_newline=True
    )
    env.filters['capfirst'] = capfirst
    env.filters['prepend'] = prepend

    try:
        template = env.get_template(template_name + ".template")
    except jinja2.TemplateNotFound:
        raise jinja2.TemplateNotFound(name=template_name,
            message="Found no template for '%s'." % template_name)
    except Exception as e:
        raise Exception("Could not read template for '%s': %s"
                        % (template_name, e))

    try:
        rendered_template = template.render(**kwargs)
    except Exception as e:
        raise Exception("Template for '%s' failed: %s" % (template_name, e))
    return rendered_template


def capfirst(value):
    """Capitalize the first character of the value."""
    return value and value[0].upper() + value[1:]


def prepend(value, prevalue="", first=False):
    """Prepend prevalue to each line of the value."""
    if first:
        return prevalue + value.replace("\n", "\n" + prevalue)
    else:
        return value.replace("\n", "\n" + prevalue)


def write_result(result, force_overwrite, verbose=0):
    """Write files.

    Parameters
    ----------
    result : dict
        Contains file names as keys and content as values

    force_overwrite : bool, optional (default: False)
        Overwrite existing file

    verbose : int, optional (default: 0)
        Verbosity level

    Returns
    -------
    written_files : list
        Names of the files that have been written
    """
    written_files = []
    msg_lines = []
    for filename, content in result.items():
        if os.path.exists(filename):
            msg_lines.append("File '%s' exists already." % filename)
            if force_overwrite:
                msg_lines[-1] += " Overwriting."
            else:
                msg_lines[-1] += " Not written."

            write = force_overwrite
        else:
            write = True

        if write:
            with open(filename, "w") as f:
                f.write(content)
            written_files.append(filename)

    if verbose >= 1 and msg_lines:
        print(os.linesep.join(msg_lines))

    return written_files
