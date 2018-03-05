import warnings
import os
import jinja2
from pkg_resources import resource_filename


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
    def __init__(self, typename):
        self.typename = typename

    @classmethod
    def handles(cls, typename, cdffpath):
        return typename+'.h' in cls.search_asn1_type(cdffpath)

    def include(self):
        """C++ include header."""
        return self.typename+".h"

    def cython_type(self):
        return "_cdff_types." + self.typename

    def python_type(self):
        return "cdff_types." + self.typename

    def copy_on_assignment(self):
        return False

    @classmethod
    def search_asn1_type(cls, cdffpath):
        """Search generated ASN1 types."""
        types_path = os.path.join(cdffpath,"Common/Types/C/")
        return os.listdir(types_path)

    def has_cdfftype(self):
        return True

class TypeRegistry(object):  # TODO global, read from config files
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
            raise NotImplementedError("No type info for '%s' available."
                                      % typename) # this error would never be triggered since DefaultTypeInfo with always return true


def write_dfn(node, output, source_folder=".", python_folder="python",
              cdffpath="CDFF"):
    """Generate code templates for a data fusion node (DFN).

    Parameters
    ----------
    node : dict
        Node configuration loaded from node description file

    output : str
        Path to output directory

    source_folder : str, optional (default: '.')
        Subdirectory of the output directory that will contain the source code
        template

    python_folder : str, optional (default: 'python')
        Subdirectory of the output directory that will contain the Python
        bindings

    cdffpath : str, optional (default: 'CDFF')
        Path to CDFF
    """
    node = validate(node)

    type_registry = TypeRegistry(cdffpath)
    src_dir = os.path.join(output, source_folder)
    python_dir = os.path.join(output, python_folder)

    if not os.path.exists(src_dir):
        os.makedirs(src_dir)
    if not os.path.exists(python_dir):
        os.makedirs(python_dir)
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


def validate(node):
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

    if "input_ports" not in node:
        validated_node["input_ports"] = []

    if "output_ports" not in node:
        validated_node["output_ports"] = []

    if "implementations" not in node:
        validated_node["implementations"] = [node["name"]]
    # TODO validate each input and output port: name and type

    return validated_node


class DFNDescriptionException(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)


def write_class(node, type_registry, template_base, class_name,
                target_folder="src", force_overwrite=False):
    result = {}

    declaration_filename = "%s.hpp" % class_name
    definition_filename = "%s.cpp" % class_name

    includes = set()
    for port in node["input_ports"] + node["output_ports"]:
        includes.add(type_registry.get_info(port["type"]).include())

    node_base_declaration = render(
        "%s.hpp" % template_base, node=node, includes=includes,
        class_name=class_name)
    target = os.path.join(target_folder, declaration_filename)
    result[target] = node_base_declaration

    node_base_definition = render(
        "%s.cpp" % template_base, declaration_filename=declaration_filename,
        node=node, class_name=class_name)
    target = os.path.join(target_folder, definition_filename)
    result[target] = node_base_definition

    return write_result(result, force_overwrite)


def write_cython(node, type_registry, template_base,
                 target_folder="python_binding"):
    result = {}

    pxd_filename = "dfn_ci_%s.pxd" % (node["name"].lower())
    _pxd_filename = "_dfn_ci_%s.pxd" % (node["name"].lower())
    pyx_filename = "dfn_ci_%s.pyx" % (node["name"].lower())

    import_cdfftypes = False
    for port in node["input_ports"] + node["output_ports"]:
        if type_registry.get_info(port["type"]).has_cdfftype():
            import_cdfftypes = True

    pxd_file = render(
        "%s.pxd" % template_base, node=node)
    target = os.path.join(target_folder, pxd_filename)
    result[target] = pxd_file

    _pxd_file = render(
        "_%s.pxd" % template_base, node=node, type_registry=type_registry,
        import_cdfftypes=import_cdfftypes)
    target = os.path.join(target_folder, _pxd_filename)
    result[target] = _pxd_file

    pyx_file = render(
        "%s.pyx" % template_base, node=node, type_registry=type_registry,
        import_cdfftypes=import_cdfftypes)
    target = os.path.join(target_folder, pyx_filename)
    result[target] = pyx_file

    return write_result(result, True)


def render(template, **kwargs):
    """Render Jinja2 template.

    Parameters
    ----------
    template : str
        name of the template, files will be searched in
        $PYTHONPATH/cdff_dev/templates/<template>.template
    kwargs : keyword arguments
        arguments that will be passed to the template

    Returns
    -------
    rendered_template : str
        template with context filled in
    """
    template_filename = resource_filename(
        "cdff_dev", os.path.join("templates", template + ".template"))
    if not os.path.exists(template_filename):
        raise IOError("No template for '%s' found." % template)
    with open(template_filename, "r") as template_file:
        template = jinja2.Template(template_file.read())
    return template.render(**kwargs)


def write_result(result, force_overwrite, verbose=0):
    written_files = []
    msg = ""
    for filename, content in result.items():
        if os.path.exists(filename):
            if msg:
                msg += os.linesep
            msg += "File '%s' exists already." % filename
            if force_overwrite:
                msg += " Overwriting."
            else:
                msg += " Not written."

            write = force_overwrite
        else:
            write = True

        if write:
            with open(filename, "w") as f:
                f.write(content)
            written_files.append(filename)

    if verbose >= 1 and msg:
        print(msg)

    return written_files
