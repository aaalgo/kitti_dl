CXXFLAGS += -std=c++14 #-I/usr/lib/x86_64-linux-gnu/hdf5/serial/include
LDFLAGS += #-L/usr/lib/x86_64-linux-gnu/hdf5/serial/lib
LDLIBS += #-lhdf5_hl -lhdf5 -lglog -lgflags

all:	eval
