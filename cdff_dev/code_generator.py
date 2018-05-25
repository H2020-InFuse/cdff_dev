import warnings
import os
import jinja2
import glob


# TODO refactor: move type info to some other file
class BasicTypeInfo(object):
    BASICTYPES = ["int", "double", "bool"]
    """Information about basic C++ types."""
    def __init__(self, typename):
        self.typename = typename

    @classmethod
    def handles(cls, typename, cdffpath):
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


class DefaultTypeInfo(object):
    """Handles any type."""
    def __init__(self, typename):
        self.typename = typename
        warnings.warn(
            "Typename '%s' is not known. It is probably not handled "
            "correctly. You might have to add an include to the C++ template "
            "and you might have to change the Python wrapper." % self.typename)

    @classmethod
    def handles(cls, typename, cdffpath):
        return True

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


class ASN1TypeInfo(object):
    """Information about ASN1 types."""
    asn1_types = {}
    def __init__(self, typename):
        self.typename = typename

    @classmethod
    def handles(cls, typename, cdffpath):
        if typename[:7] == 'asn1Scc':
            return typename[7:] in cls._search_asn1_type(cdffpath)

    def include(self):
        """C++ include header."""
        for key in self.asn1_types.keys():
            if self.typename[7:] in self.asn1_types[key]:
                return key.split('.')[0] + ".h"

    def cython_type(self):
        return "_cdff_types." + self.typename

    def python_type(self):
        return "cdff_types." + self.typename[7:]

    def copy_on_assignment(self):
        return False

    @classmethod
    def _search_asn1_type(cls, cdffpath):
        """Search generated ASN1 types."""
        ASN1_paths = glob.glob(os.path.join(
            cdffpath, "Common/Types/ASN.1/ESROCOS/*/*.asn")) + glob.glob(
            os.path.join(cdffpath, "Common/Types/ASN.1/InFuse/*.asn"))
        asn1_types = {}
        asn1_list = []
        for asn1_path in ASN1_paths:
            with open(asn1_path, "r", encoding="utf8") as f:
                file_read = f.read()
            splitted_file = file_read.replace('\n', ' ').split('::=')[:-1]
            types_in_file = []
            for i,f in enumerate(splitted_file):
                asn1_type = list(filter(lambda a: a != '', f.split(' ')))[-1]
                if asn1_type != "DEFINITIONS" and asn1_type not in asn1_list:
                    asn1_list.append(asn1_type)
                    types_in_file.append(asn1_type)
            asn1_types[asn1_path.split('/')[-1]] = types_in_file
        cls.asn1_types = asn1_types
        return asn1_list

    def has_cdfftype(self):
        return True


class TypeRegistry(object):
    TYPEINFOS = [BasicTypeInfo, ASN1TypeInfo, DefaultTypeInfo]
    """Registry for InFuse type information."""
    def __init__(self, cdffpath):
        self.cache = {}
        self.cdffpath = cdffpath

    def get_info(self, typename):
        for TypeInfo in self.TYPEINFOS:
            type_found = TypeInfo.handles(typename, self.cdffpath)
            if type_found:
                if typename not in self.cache:
                    self.cache[typename] = TypeInfo(typename)
                return self.cache[typename]
        else:
            # this error would never be triggered since DefaultTypeInfo with
            # always return true
            raise NotImplementedError("No type info for '%s' available."
                                      % typename)


class DFNDescriptionException(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)


class DFPCDescriptionException(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)


class PortDescriptionException(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)


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

    type_registry = TypeRegistry(cdffpath)
    src_dir, python_dir = _prepare_output_directory(
        output, source_folder, python_folder)
    interface_files = write_class(
        node, type_registry, "Interface",
        "%sInterface" % node["name"], target_folder=src_dir,
        force_overwrite=True)
    implementation_files = []
    for implementation in node["implementations"]:
        implementation_files.extend(
            write_class(node, type_registry, "Node", implementation,
                        target_folder=src_dir))
    cython_files = write_cython(node, type_registry, "Node",
                                target_folder=python_dir)
    return interface_files + implementation_files + cython_files


def write_dfpc(dfpc, cdffpath,
               output, source_folder=".", python_folder="python"):
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

    type_registry = TypeRegistry(cdffpath)
    src_dir, python_dir = _prepare_output_directory(
        output, source_folder, python_folder)
    interface_files = write_class(
        dfpc, type_registry, "DFPCInterface",
        "%sInterface" % dfpc["name"], target_folder=src_dir,
        force_overwrite=True)
    implementation_files = write_class(
        dfpc, type_registry, "DFPCImplementation", dfpc["name"],
        target_folder=src_dir)
    dfpc["implementations"] = [dfpc["name"]]
    cython_files = write_cython(dfpc, type_registry, "DFPC",
                                target_folder=python_dir)
    return interface_files + implementation_files + cython_files


def validate_node(node):
    """Validate node description.

    Raises a DFNDescriptionException if validation is not
    successful.

    A validated node contains:
    - name
    - input_ports
    - output_ports
    - implementations

    Parameters
    ----------
    node : dict
        Node description

    Returns
    -------
    validated_node : dict
        Validated node description
    """
    validated_node = {}
    validated_node.update(node)

    if "name" not in node:
        raise DFNDescriptionException(
            "DFN description has no attribute 'name'.")

    _validate_ports(validated_node)

    if "implementations" not in node:
        validated_node["implementations"] = [node["name"]]

    return validated_node


def validate_dfpc(dfpc):
    """Validate DFPC description.

    Raises a DFNDescriptionException if validation is not
    successful.

    A validated DFPC contains:
    - name
    - input_ports
    - output_ports
    - ...

    Parameters
    ----------
    dfpc : dict
        DFPC description

    Returns
    -------
    validated_dfpc : dict
        Validated DFPC description
    """
    validated_dfpc = {}
    validated_dfpc.update(dfpc)

    if "name" not in dfpc:
        raise DFPCDescriptionException(
            "DFPC description has no attribute 'name'.")

    _validate_ports(validated_dfpc)
    _validate_dfpc_port_connections(validated_dfpc)
    _validate_dfpc_operations(validated_dfpc)
    _validate_dfpc_internal_connections(validated_dfpc)

    return validated_dfpc


def _validate_ports(desc):
    if "input_ports" not in desc:
        desc["input_ports"] = []

    if "output_ports" not in desc:
        desc["output_ports"] = []

    for port in desc["input_ports"] + desc["output_ports"]:
        if "name" not in port:
            raise PortDescriptionException("Port has no name: %s" % port)
        if "type" not in port:
            raise PortDescriptionException("Port has no type: %s" % port)


def _validate_dfpc_port_connections(desc):
    for port in desc["input_ports"] + desc["output_ports"]:
        if "connections" not in port or len(port["connections"]) == 0:
            raise PortDescriptionException(
                "Port has no connections: %s" % port)


def _validate_dfpc_operations(desc):
    if "operations" not in desc:
        desc["operations"] = []
        return

    for op in desc["operations"]:
        if "name" not in op:
            raise DFPCDescriptionException("Operation has no name: %s" % op)
        if "output_type" not in op:
            op["output_type"] = "void"

        if "inputs" not in op:
            op["inputs"] = []
        for inp in op["inputs"]:
            if "name" not in inp:
                raise DFPCDescriptionException(
                    "Input of operation '%s' has no name: %s"
                    % (op["name"], inp))
            if "type" not in inp:
                raise DFPCDescriptionException(
                    "Input '%s' of operation '%s' has no type"
                    % (inp["name"], op["name"]))


def _validate_dfpc_internal_connections(desc):
    if "internal_connections" not in desc:
        desc["internal_connections"] = []

    for conn in desc["internal_connections"]:
        if "from" not in conn:
            raise DFPCDescriptionException(
                "No 'from' in connection: %s" % conn)
        if "to" not in conn:
            raise DFPCDescriptionException("No 'to' in connection: %s" % conn)

        for port in [conn["from"], conn["to"]]:
            if "dfn" not in port:
                raise DFPCDescriptionException(
                    "No DFN is specified in connection: %s" % conn)
            if "port" not in port:
                raise DFPCDescriptionException(
                    "No port is specified in connection: %s" % conn)


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
    for port in desc["input_ports"] + desc["output_ports"]:
        includes.add(type_registry.get_info(port["type"]).include())

    base_declaration = render(
        "%s.hpp" % template_base, desc=desc, includes=includes,
        class_name=class_name)
    target = os.path.join(target_folder, declaration_filename)
    result[target] = base_declaration

    base_definition = render(
        "%s.cpp" % template_base, declaration_filename=declaration_filename,
        desc=desc, class_name=class_name)
    target = os.path.join(target_folder, definition_filename)
    result[target] = base_definition

    return write_result(result, force_overwrite)


def write_cython(desc, type_registry, template_base,
                 target_folder="python", namespace_prefix="dfn_ci"):
    """Write Python binding based on Cython.

    Parameters
    ----------
    desc : dict
        Component description

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
        "Declaration.pxd", desc=desc,
        namespace_prefix=namespace_prefix)
    target = os.path.join(target_folder, pxd_filename)
    result[target] = pxd_file

    _pxd_file = render(
        "_%s.pxd" % template_base, desc=desc, type_registry=type_registry,
        import_cdfftypes=import_cdfftypes)
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
        "%s.pyx" % template_base, desc=desc, import_cdfftypes=import_cdfftypes,
        input_ports=input_ports, output_ports=output_ports,
        operations=operations)

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
        keep_trailing_newline=True
    )
    env.filters['capfirst'] = capfirst

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
