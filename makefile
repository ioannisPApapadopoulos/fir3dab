
double-pipe-smoothed-bcs-tables:
	cd examples/double-pipe-smoothed-bcs-tables && python3 double-pipe-bdm.py
	cd examples/double-pipe-smoothed-bcs-tables && python3 double-pipe-th.py
	cd examples/double-pipe-smoothed-bcs-tables && python3 table-double-pipe.py
	cd examples/double-pipe-smoothed-bcs-tables && python3 convergence-plots.py

3d-5-holes-coarse:
	cd examples/3d-5-holes && mpiexec -n 16 python3 3d-5-holes.py

3d-5-holes-prolong-solutions:
	cd examples/3d-5-holes && mpiexec -n 16 python3 make_vtu.py
	cd examples/3d-5-holes && mpiexec -n 32 python3 save_h5.py

3d-5-holes-fine:
	cd examples/3d-5-holes && mpiexec -n 32 python3 mg-3d-5-holes.py

3d-5-holes-3-grid:
	cd examples/3d-5-holes && mpiexec -n 32 python3 mg-3d-5-holes-3grid.py

3d-cross-channel-coarse:
	cd examples/3d-cross-channel && mpiexec -n 16 python3 3d-cross-channel.py

3d-cross-channel-prolong-solutions:
	cd examples/3d-cross-channel && mpiexec -n 16 python3 make_vtu.py
	cd examples/3d-cross-channel && mpiexec -n 32 python3 save_h5.py

3d-cross-channel-fine:
	cd examples/3d-cross-channel && mpiexec -n 32 python3 mg-3d-cross-channel.py

double-pipe-p-refinement-interpolate-solutions:
	cd examples/double-pipe/interpolation-scripts && python3 make_vtu.py
	cd examples/double-pipe/interpolation-scripts && mpiexec -n 4 python3 save_h5-2grid.py
	cd examples/double-pipe/interpolation-scripts && mpiexec -n 11 python3 save_h5-3grid.py

double-pipe-tables:
	cd examples/double-pipe-tables && python3 double-pipe-bdm.py
	cd examples/double-pipe-tables && python3 double-pipe-th.py
	cd examples/double-pipe-tables && python3 table-double-pipe.py
	cd examples/double-pipe-tables && python3 convergence-plots.py
