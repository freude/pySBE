
compile :
	rm -f *.so
	rm -f ../sbe/*.so
	f2py3 -c P_loop.f90 -m P_loop
	f2py3 -c fft_loop.f90 -m fft_loop
	cp *.so ../sbe/
clean: 
	rm -f *.so
	rm -f ../sbe/*.so

