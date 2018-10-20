all:cutline_plus.so cutline_plus
cutline_plus.so:cutline_plus.cpp
	 g++ -fpic -shared $< -o $@ `pkg-config opencv --cflags --libs`
cutline_plus:cutline_plus.cpp
	g++ -fpic $< -o $@ `pkg-config opencv --cflags --libs`
clean:
	rm *.so	
