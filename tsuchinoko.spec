# -*- mode: python ; coding: utf-8 -*-
import os
import glob
import sys

import dask
import distributed
import event_model
import debugpy
import pyqode.python.backend

from tsuchinoko import assets, examples
import tsuchinoko

block_cipher = None

# Include assets
datas_src = [path for path in glob.glob(os.path.join(assets.__path__[0], "**/*.*"), recursive=True) if "__init__.py" not in path]
datas_dst = [os.path.dirname(os.path.relpath(path, os.path.join(tsuchinoko.__path__[0], os.path.pardir))) or '.' for path in datas_src]

# Dask needs its config yaml
datas_src.append(os.path.join(dask.__path__[0], '*.yaml'))
datas_dst.append('dask')

# Distributed needs its yaml
datas_src.append(os.path.join(distributed.__path__[0], 'distributed.yaml'))
datas_dst.append('distributed')

# event_model needs its json
jsons = glob.glob(os.path.join(event_model.__path__[0], 'schemas/*.json'))
datas_src.extend(jsons)
datas_dst.extend('event_model/schemas' for path in jsons)

# examples data image
images = glob.glob(os.path.join(examples.__path__[0], '*.jpg'))
images += glob.glob(os.path.join(examples.__path__[0], '*.png'))
datas_src.extend(images)
datas_dst.extend('tsuchinoko/examples' for path in images)

# include example scripts
examples = glob.glob(os.path.join(examples.__path__[0], '*.py'))
datas_src.extend(examples)
datas_dst.extend('tsuchinoko/examples' for path in examples)

# include debugpy
datas_src.append(debugpy.__path__[0])
datas_dst.append("debugpy")

# pyqode server pyqode\\python\\backend\\server.py
datas_src.append(os.path.join(pyqode.python.backend.__path__[0], '*.py'))
datas_dst.append("pyqode/python/backend/")

# functorch and torch (mac only)
#if sys.platform == 'darwin':
#    functorch = glob.glob(os.path.join(functorch.__path__[0], '.dylibs', '*.dylib'))
#    datas_src.extend(functorch)
#    datas_dst.extend('functorch/.dylibs' for dylib in functorch)

#    torch = glob.glob(os.path.join(torch.__path__[0], 'lib', '*.dylib'))
#    datas_src.extend(torch)
#    datas_dst.extend('torch/lib' for dylib in torch)

print('extras:')
print(list(zip(datas_src, datas_dst)))

a = Analysis(
    [os.path.join('tsuchinoko','examples','client_demo.py')],
    pathex=[],
    binaries=[],
    datas=zip(datas_src, datas_dst),
    hiddenimports=['tsuchinoko.graphs.common',
                   'tsuchinoko.examples.adaptive_demo',
                   'tsuchinoko.examples.grid_demo',
                   'tsuchinoko.examples.high_dimensionality_server_demo',
                   'tsuchinoko.examples.multi_task_server_demo',
                   'tsuchinoko.examples.quadtree_demo',
                   'tsuchinoko.examples.server_demo',
                   'tsuchinoko.examples.server_demo_bluesky',
                   'tsuchinoko.examples.vector_metric_demo',
                   'event_model'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

a2 = Analysis(
    [os.path.join('tsuchinoko','examples','_launch_demo.py')],
    pathex=[],
    binaries=[],
    datas=zip(datas_src, datas_dst),
    # hiddenimports=['tsuchinoko.graphs.common',
    #                'tsuchinoko.examples.adaptive_demo',
    #                'tsuchinoko.examples.grid_demo',
    #                'tsuchinoko.examples.high_dimensionality_server_demo',
    #                'tsuchinoko.examples.multi_task_server_demo',
    #                'tsuchinoko.examples.quadtree_demo',
    #                'tsuchinoko.examples.server_demo',
    #                'tsuchinoko.examples.server_demo_bluesky',
    #                'tsuchinoko.examples.vector_metric_demo',],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

a3 = Analysis(
    [os.path.join('tsuchinoko','_bootstrap.py')],
    pathex=[],
    binaries=[],
    datas=zip(datas_src, datas_dst),
    # hiddenimports=['tsuchinoko.graphs.common',
    #                'tsuchinoko.examples.adaptive_demo',
    #                'tsuchinoko.examples.grid_demo',
    #                'tsuchinoko.examples.high_dimensionality_server_demo',
    #                'tsuchinoko.examples.multi_task_server_demo',
    #                'tsuchinoko.examples.quadtree_demo',
    #                'tsuchinoko.examples.server_demo',
    #                'tsuchinoko.examples.server_demo_bluesky',
    #                'tsuchinoko.examples.vector_metric_demo',],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

pyz2 = PYZ(a2.pure, a2.zipped_data, cipher=block_cipher)

pyz3 = PYZ(a3.pure, a3.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='tsuchinoko_client',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=assets.path('tsuchinoko.png')
)

exe2 = EXE(
    pyz2,
    a2.scripts,
    [],
    exclude_binaries=True,
    name='tsuchinoko_demo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=assets.path('tsuchinoko.png')
)

exe3 = EXE(
    pyz3,
    a3.scripts,
    [],
    exclude_binaries=True,
    name='tsuchinoko_bootstrap',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=assets.path('tsuchinoko.png')
)

print(f'platform: {sys.platform}')
if sys.platform == 'win32':
    coll = COLLECT(
        exe, exe2, exe3,
        a.binaries, a2.binaries, a3.binaries,
        a.zipfiles, a2.zipfiles, a3.zipfiles,
        a.datas, a2.datas, a3.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='Tsuchinoko',
    )
elif sys.platform == 'darwin':
    app = BUNDLE(
        exe, exe2, exe3,
        a.binaries, a2.binaries, a3.binaries,
        a.zipfiles, a2.zipfiles, a3.zipfiles,
        a.datas, a2.datas, a3.datas,
        name='Tsuchinoko',
        icon=assets.path('tsuchinoko.icns'),
        bundle_identifier=None,
    )


