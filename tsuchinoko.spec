# -*- mode: python ; coding: utf-8 -*-
import os
import glob

import dask
import distributed
import event_model

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
datas_src.append(os.path.join(examples.__path__[0], 'sombrero_pug.jpg'))
datas_dst.append('tsuchinoko/examples')
datas_src.append(os.path.join(examples.__path__[0], 'peak1.png'))
datas_dst.append('tsuchinoko/examples')
datas_src.append(os.path.join(examples.__path__[0], 'peak2.png'))
datas_dst.append('tsuchinoko/examples')

print('extras:')
print(list(zip(datas_src, datas_dst)))

a = Analysis(
    ['tsuchinoko\\examples\\client_demo.py'],
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
                   'tsuchinoko.examples.vector_metric_demo'],
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
    ['tsuchinoko\\examples\\_launch_demo.py'],
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
                   'tsuchinoko.examples.vector_metric_demo'],
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

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Tsuchinoko',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=assets.path('tsuchinoko.png')
)

coll = COLLECT(
    exe, exe2,
    a.binaries, a2.binaries,
    a.zipfiles, a2.zipfiles,
    a.datas, a2.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Tsuchinoko',
)
