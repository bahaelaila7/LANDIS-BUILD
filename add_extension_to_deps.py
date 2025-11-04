#!/usr/bin/env python
import json
import argparse
import pathlib
import sys
import pefile


def get_dll_meta_data(dll_path):
    pe = pefile.PE(dll_path)
    result = {}
    if hasattr(pe, "FileInfo"):
        for fileinfo in pe.FileInfo:
            for subfileinfo in fileinfo:
                if subfileinfo.Key == b"StringFileInfo":
                    for st in subfileinfo.StringTable:
                        for key, value in st.entries.items():
                            result[key.decode()] = value.decode(errors="ignore")
    return {
        "FileVersion": result.get("FileVersion"),
        "ProductVersion": result.get("ProductVersion"),
        "CompanyName": result.get("CompanyName"),
        "OriginalFilename": result.get("OriginalFilename"),
        # sometimes included
        "AssemblyVersion": result.get("Assembly Version") or None,
    }


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        "Adds Extension DLL to Landis.Console.deps.json"
    )
    argparser.add_argument("-j", "--json-path", type=pathlib.Path)
    argparser.add_argument("-a", "--assembly-name", type=str, default=None)
    argparser.add_argument("-d", "--extension-dll", type=pathlib.Path)
    argparser.add_argument("-nv", "--nuget-package-version", type=str, default=None)
    args = argparser.parse_args()
    if not args.json_path.exists():
        raise Exception(f"The file {args.json_path} does not exist")
    if not args.extension_dll.exists():
        raise Exception(f"The file {args.extension_dll} does not exist")

    dll_metadata = get_dll_meta_data(args.extension_dll)

    with open(args.json_path, "r") as jf:
        json_dict = json.load(jf)
    assembly_name = args.assembly_name or pathlib.Path(args.extension_dll.name).stem
    dll_version = dll_metadata["FileVersion"] or "1.0.0"
    assembly_version = dll_metadata["AssemblyVersion"] or dll_version
    nuget_package_version = args.nuget_package_version or assembly_version
    assembly_canonical_name = f"{assembly_name}/{nuget_package_version}"
    dll_rel_path = str(args.extension_dll.relative_to(args.json_path.parent))
    target_entry = {
        "runtime": {
            dll_rel_path: {
                "assemblyVersion": assembly_version,
                "fileVersion": dll_version,
            }
        }
    }
    library_entry = {
        "type": "reference",
        "serviceable": False,
        "sha512": "",
        "path": dll_rel_path,
    }

    found_in_targets = []
    for target, outs in json_dict["targets"].items():
        target_to_remove = []
        for out in outs:
            if out.startswith("Landis.Console/"):
                found_in_targets.append(target)
                to_remove = []
                if "dependencies" not in outs[out]:
                    outs[out]["dependencies"] = dict()
                out_deps = outs[out]["dependencies"]
                for dep in out_deps:
                    if dep == assembly_name:
                        to_remove.append(dep)
                for dep in to_remove:
                    del out_deps[dep]
                out_deps[assembly_canonical_name] = assembly_version
            elif out.startswith(f"{assembly_name}/"):
                target_to_remove.append(out)
        for out in target_to_remove:
            del outs[out]

    libs_to_remove = []
    for lib in json_dict["libraries"]:
        if lib.startswith(f"{assembly_name}/"):
            libs_to_remove.append(lib)
    for lib in libs_to_remove:
        del json_dict["libraries"][lib]

    if len(found_in_targets) == 0:
        raise Exception(
            f"Json file {args.json_path} does not list target Landis.Console"
        )
    for target in found_in_targets:
        json_dict["targets"][target][assembly_canonical_name] = target_entry

    json_dict["libraries"][assembly_canonical_name] = library_entry

    with open(args.json_path, "w") as jf:
        json.dump(json_dict, jf, indent=4)
