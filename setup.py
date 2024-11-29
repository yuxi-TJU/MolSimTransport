from setuptools import setup, find_packages

setup(
    name='MolSimTransport',
    version='1.0.2',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'rdkit',
        'tblite==0.3.0',
        'importlib.resources',
    ],
    entry_points={
        'console_scripts': [
            "MolSimTransport=MolSimTransport.main:main",
            'L1_EHT=MolSimTransport.l1_bare_mol_eht:run_eht_wba',
            'L1_XTB=MolSimTransport.l1_bare_mol_xtb:run_xtb_wba',
            'L2_Align=MolSimTransport.l2_em_align:main',
            'L2_Trans=MolSimTransport.l2_em_trans:main',
            'L2_MPSH=MolSimTransport.l2_generate_mpsh_molden:main',
            'L3_Trans=MolSimTransport.l3_junction_trans:main',
            'L3_EEF=MolSimTransport.l3_junction_with_eef:main',
            'L3_MPSH=MolSimTransport.l3_generate_mpsh_molden:main',
            'L3_EC=MolSimTransport.l3_trans_eigenstate:main',
        ],
    },
    
    package_data={
        'MolSimTransport.utils': ['*.json'], 
        # 'share': ['*.xyz','*.dat', '*.mat'],
    },

    author='Xi Yu, Qiang Qi and Xuan Ji',
    author_email='xi.yu@tju.edu.cn',
    description='A python package for quickly calculating the transport properties of molecular junctions at multiple scales',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    url='https://github.com/yuxi-TJU/MolSim-Transport',

    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
    ],
    python_requires='>=3.9',
)
