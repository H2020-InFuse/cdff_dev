import os
import jinja2
from pkg_resources import resource_filename


class TypeRegistry(object):  # TODO global, read from config files
    """Registry for InFuse type information."""
    def __init__(self):
        self.cache = {}

    def get_info(self, typename):
        if typename in BasicTypeInfo.BASICTYPES:
            if typename not in self.cache:
                self.cache[typename] = BasicTypeInfo(typename)
            return self.cache[typename]
        else:
            raise NotImplementedError("No type info for '%s' available."
                                      % typename)


class BasicTypeInfo(object):
    BASICTYPES = ["int", "double", "bool"]
    """Information about basic C++ types."""
    def __init__(self, typename):
        self.typename = typename

    def include(self):
        """C++ include header."""
        return None

    def cython_type(self):
        return self.typename

    def python_type(self):
        return self.typename

    def copy_on_assignment(self):
        return True


def write_dfn(node, output, cdffpath):
    type_registry = TypeRegistry()
    src_dir = os.path.join(output, "src")
    python_dir = os.path.join(output, "python")
    if not os.path.exists(src_dir):
        os.makedirs(src_dir)
    if not os.path.exists(python_dir):
        os.makedirs(python_dir)
    interface_files = write_class(
        node, type_registry, "Interface", "Interface", target_folder=src_dir,
        force_overwrite=True)
    implementation_files = write_class(
        node, type_registry, "Node", "", target_folder=src_dir)
    cython_files = write_cython(node, type_registry, "Node",
                                target_folder=python_dir)
    return interface_files + implementation_files + cython_files


def write_class(node, type_registry, template_base, file_suffix,
                target_folder="src", force_overwrite=False):
    result = {}

    declaration_filename = "%s%s.hpp" % (node["name"], file_suffix)
    definition_filename = "%s%s.cpp" % (node["name"], file_suffix)

    includes = set()
    for input_port in node["input_ports"]:
        includes.add(type_registry.get_info(input_port["type"]).include())
    for output_port in node["output_ports"]:
        includes.add(type_registry.get_info(output_port["type"]).include())

    node_base_declaration = render(
        "%s.hpp" % template_base, node=node, includes=includes)
    target = os.path.join(target_folder, declaration_filename)
    result[target] = node_base_declaration

    node_base_definition = render(
        "%s.cpp" % template_base, declaration_filename=declaration_filename,
        node=node)
    target = os.path.join(target_folder, definition_filename)
    result[target] = node_base_definition

    return write_result(result, force_overwrite)


def write_cython(node, type_registry, template_base,
                 target_folder="python_binding"):
    result = {}

    pxd_filename = "dfn_ci_%s.pxd" % (node["name"].lower())
    _pxd_filename = "_dfn_ci_%s.pxd" % (node["name"].lower())
    pyx_filename = "dfn_ci_%s.pyx" % (node["name"].lower())

    pxd_file = render(
        "%s.pxd" % template_base, node=node)
    target = os.path.join(target_folder, pxd_filename)
    result[target] = pxd_file

    _pxd_file = render(
        "_%s.pxd" % template_base, node=node)
    target = os.path.join(target_folder, _pxd_filename)
    result[target] = _pxd_file

    pyx_file = render(
        "%s.pyx" % template_base, node=node,
        type_registry=type_registry)
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
