#!/usr/bin/env bash
# Expose the USD and PhysX schema modules from an Isaac Sim 5.1 pip/conda install.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 PYTHON_SCRIPT [ARGS ...]" >&2
    exit 2
fi

isaac_python="${ISAAC_PYTHON:-python3}"
python_prefix="$(${isaac_python} -c 'import sys; print(sys.prefix)')"
isaacsim_dir="$(${isaac_python} -c 'import importlib.util, pathlib; spec = importlib.util.find_spec("isaacsim"); assert spec and spec.submodule_search_locations, "isaacsim package not found"; print(pathlib.Path(next(iter(spec.submodule_search_locations))))')"
extcache_dir="${isaacsim_dir}/extscache"
usd_libs_dir="$(find "${extcache_dir}" -maxdepth 1 -type d -name 'omni.usd.libs-*' -print -quit)"
physx_schema_dir="$(find "${extcache_dir}" -maxdepth 1 -type d -name 'omni.usd.schema.physx-*' -print -quit)"

if [[ -z "${usd_libs_dir}" || -z "${physx_schema_dir}" ]]; then
    echo "Could not locate Isaac Sim USD/PhysX schema extensions under ${extcache_dir}" >&2
    exit 1
fi

plugin_path=""
for resources_dir in "${physx_schema_dir}"/plugins/*/resources; do
    if [[ -d "${resources_dir}" ]]; then
        plugin_path="${plugin_path:+${plugin_path}:}${resources_dir}"
    fi
done

export PYTHONPATH="${usd_libs_dir}:${physx_schema_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${python_prefix}/lib:${usd_libs_dir}/bin:${physx_schema_dir}/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PXR_PLUGINPATH_NAME="${plugin_path}${PXR_PLUGINPATH_NAME:+:${PXR_PLUGINPATH_NAME}}"

exec "${isaac_python}" "$@"
