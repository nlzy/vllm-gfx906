#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#
# A command line tool for running pytorch's hipify preprocessor on CUDA
# source files.
#
# See https://github.com/ROCm/hipify_torch
# and <torch install dir>/utils/hipify/hipify_python.py
#

import argparse
import os
import shutil

from torch.utils.hipify.hipify_python import hipify

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Project directory where all the source + include files live.
    parser.add_argument(
        "-p",
        "--project_dir",
        help="The project directory.",
    )

    # Directory where hipified files are written.
    parser.add_argument(
        "-o",
        "--output_dir",
        help="The output directory.",
    )

    # Source files to convert.
    parser.add_argument("sources",
                        help="Source files to hipify.",
                        nargs="*",
                        default=[])

    args = parser.parse_args()

    # Limit include scope to project_dir only
    includes = [os.path.join(args.project_dir, '*')]

    # Get absolute path for all source files.
    extra_files = [os.path.abspath(s) for s in args.sources]

    # Copy sources from project directory to output directory.
    # The directory might already exist to hold object files so we ignore that.
    shutil.copytree(args.project_dir, args.output_dir, dirs_exist_ok=True)

    hipify_result = hipify(project_directory=args.project_dir,
                           output_directory=args.output_dir,
                           header_include_dirs=[],
                           includes=includes,
                           extra_files=extra_files,
                           show_detailed=True,
                           is_pytorch_extension=True,
                           hipify_extra_files_only=True)

    hipified_sources = []
    for source in args.sources:
        s_abs = os.path.abspath(source)
        hipified_s_abs = None
        if s_abs in hipify_result:
            hipified_s_abs = hipify_result[s_abs].hipified_path

        if hipified_s_abs is None:
            # hipify may decide that the source does not need any edits. However,
            # our build expects a hipified translation unit with a `.hip`
            # extension to exist in the build tree. Create one by copying the
            # original file into the mirrored build directory with the expected
            # name.
            rel_path = os.path.relpath(s_abs, args.project_dir)
            hip_rel = rel_path.replace("cuda", "hip")
            if hip_rel.endswith(".cu"):
                hip_rel = hip_rel[:-3] + ".hip"
            hip_abs = os.path.abspath(os.path.join(args.output_dir, hip_rel))
            os.makedirs(os.path.dirname(hip_abs), exist_ok=True)
            shutil.copy2(os.path.join(args.output_dir, rel_path), hip_abs)
            hipified_s_abs = hip_abs

        hipified_sources.append(hipified_s_abs)

    assert (len(hipified_sources) == len(args.sources))

    # Print hipified source files.
    print("\n".join(hipified_sources))
