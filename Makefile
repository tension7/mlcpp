COMPILER=g++-4.8
FLAGS=-std=c++11 -Wall -Wno-sign-compare -O0
LIBS=-lglog -lgflags
INCLUDE=-I.
CPP_CMD=${COMPILER} ${FLAGS} ${INCLUDE}

all: decision_tree_train decision_tree_eval

decision_tree_train:
	mkdir -p _bin/tree
	${CPP_CMD} -o _bin/$@ tree/decision_tree_train.cpp ${LIBS}

decision_tree_eval:
	mkdir -p _bin/tree
	${CPP_CMD} -o _bin/$@ tree/decision_tree_eval.cpp ${LIBS}

clean:
	rm -rf _bin
