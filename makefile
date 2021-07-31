
double-pipe-tables:
	cd examples/double-pipe-tables && python3 double-pipe-bdm.py
	cd examples/double-pipe-tables && python3 double-pipe-th.py
	cd examples/double-pipe-tables && python3 table-double-pipe.py
	cd examples/double-pipe-tables && python3 convergence-plots.py

3d-5-holes-coarse:
	cd examples/3d-5-holes && mpiexec -n 16 python3 3d-5-holes.py

3d-5-holes-prolong-solutions:
	cd examples/3d-5-holes && mpiexec -n 16 python3 make_vtu.py
	cd examples/3d-5-holes && mpiexec -n 32 python3 save_h5.py

3d-5-holes-fine:
	cd examples/3d-5-holes && mpiexec -n 32 python3 mg-3d-5-holes.py

3d-cross-channel-coarse:
	cd examples/3d-cross-channel && mpiexec -n 16 python3 3d-cross-channel.py

3d-cross-channel-prolong-solutions:
	cd examples/3d-cross-channel && mpiexec -n 16 python3 make_vtu.py
	cd examples/3d-cross-channel && mpiexec -n 32 python3 save_h5.py

3d-cross-channel-fine:
	cd examples/3d-cross-channel && mpiexec -n 32 python3 mg-3d-cross-channel.py
