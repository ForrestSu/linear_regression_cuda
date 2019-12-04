include ../Makefile.mk

ALL_CCFLAGS += -dc -rdc=true
# extra 
EXTRA_CXXFLAGS := $(addprefix -Xcompiler ,-Wall -fPIC -DOIMBALANCE_LIB)
ALL_CCFLAGS += $(EXTRA_CXXFLAGS)
ALL_LDFLAGS += $(EXTRA_CXXFLAGS)

INCLUDES  += -I/usr/include/qt/ -I../common/inc -I../public 
LIBRARIES += -L../../../bin/linux
LIBRARIES += -lpthread -lprotobuf -lzmq -luv -lhiredis -lemsproto -lzuvoutility -lQt5Core -lbaseutils
LIBRARIES += -lcublas -lcublas_device -lcudadevrt 
################################################################################
.PHONY:all clean run

SRC_CU  := $(wildcard *.cu  ../public/cuda_util.cu)
OBJ_CU  := $(SRC_CU:%.cu=%.o)
SRC_CPP := $(wildcard *.cpp ../public/*.cpp)
OBJ_CPP := $(SRC_CPP:%.cpp=%.o)
# Target rules
TARGET  = liborder_imbalance.so.1.0.0
all: build

build: $(TARGET)

%.o: %.cu 
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

%.o: %.cpp 
	$(NVCC) -std=c++11 $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(TARGET): $(OBJ_CU)  $(OBJ_CPP)
	$(NVCC) -shared $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	mkdir -p ../../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)
	cp $@ ../../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)

run: build
	sudo cp $(TARGET) /usr/lib/ems/
	sudo cp order_imbalance.xml /etc/ems/
	/usr/bin/zuvo /etc/ems/order_imbalance.xml -autostart

clean:
	rm -f $(TARGET) *.o ../public/*.o




################################################################################
# By Jerry.Liu: For install.
################################################################################

first: all

####### Install
QINSTALL_PROGRAM = /usr/bin/qmake -install qinstall -exe
QINSTALL      = /usr/bin/qmake -install qinstall

SYMLINK       = ln -f -s
DEL_FILE      = rm -f
DEL_DIR       = rmdir

TARGET        = liborder_imbalance.so.1.0.0
TARGETA       = ../../../bin/linux/liborder_imbalance.a
TARGET0       = liborder_imbalance.so
TARGETD       = liborder_imbalance.so.1.0.0
TARGET1       = liborder_imbalance.so.1
TARGET2       = liborder_imbalance.so.1.0


install_target: first FORCE
	@test -d $(INSTALL_ROOT)/usr/lib/ems || mkdir -p $(INSTALL_ROOT)/usr/lib/ems
	-$(QINSTALL_PROGRAM) ../../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)/$(TARGET) $(INSTALL_ROOT)/usr/lib/ems/$(TARGET)
	-$(SYMLINK) $(TARGET) $(INSTALL_ROOT)/usr/lib/ems/$(TARGET0)
	-$(SYMLINK) $(TARGET) $(INSTALL_ROOT)/usr/lib/ems/$(TARGET1)
	-$(SYMLINK) $(TARGET) $(INSTALL_ROOT)/usr/lib/ems/$(TARGET2)

uninstall_target: FORCE
	-$(DEL_FILE) $(INSTALL_ROOT)/usr/lib/ems/$(TARGET) 
	 -$(DEL_FILE) $(INSTALL_ROOT)/usr/lib/ems/$(TARGET0) 
	 -$(DEL_FILE) $(INSTALL_ROOT)/usr/lib/ems/$(TARGET1) 
	 -$(DEL_FILE) $(INSTALL_ROOT)/usr/lib/ems/$(TARGET2)
	-$(DEL_DIR) $(INSTALL_ROOT)/usr/lib/ems/ 


install_configures: first FORCE
	@test -d $(INSTALL_ROOT)/etc/ems/ || mkdir -p $(INSTALL_ROOT)/etc/ems/
	-$(QINSTALL) ./order_imbalance.xml $(INSTALL_ROOT)/etc/ems/order_imbalance.xml

uninstall_configures: FORCE
	-$(DEL_FILE) -r $(INSTALL_ROOT)/etc/ems/order_imbalance.xml
	-$(DEL_DIR) $(INSTALL_ROOT)/etc/ems/ 


install: install_target install_configures  FORCE

uninstall: uninstall_target uninstall_configures  FORCE

FORCE:

