//
// Created by durst on 3/1/22.
//
#include "bots/thinker.h"
#define PY_SSIZE_T_CLEAN
#include <Python.h>

int run_model(string moduleName, string functionName) {
    PyObject *pName, *pModule, *pFunc;
    PyObject *pArg, *pValue;
    int i;

    Py_Initialize();
    pName = PyUnicode_DecodeFSDefault(moduleName.c_str());
    /* Error checking of pName left out */

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    vector<string> a = {"id","tick id","round id","source player id","team","cur nav area","cur pos x","cur pos y","cur pos z","cur nav 0 friends","cur nav 0 enemies","cur nav 1 friends","cur nav 1 enemies","cur nav 2 friends","cur nav 2 enemies","cur nav 3 friends","cur nav 3 enemies","cur nav 4 friends","cur nav 4 enemies","cur nav 5 friends","cur nav 5 enemies","cur nav 6 friends","cur nav 6 enemies","cur nav 7 friends","cur nav 7 enemies","cur nav 8 friends","cur nav 8 enemies","cur nav 9 friends","cur nav 9 enemies","cur nav 10 friends","cur nav 10 enemies","last nav area","last pos x","last pos y","last pos z","last nav 0 friends","last nav 0 enemies","last nav 1 friends","last nav 1 enemies","last nav 2 friends","last nav 2 enemies","last nav 3 friends","last nav 3 enemies","last nav 4 friends","last nav 4 enemies","last nav 5 friends","last nav 5 enemies","last nav 6 friends","last nav 6 enemies","last nav 7 friends","last nav 7 enemies","last nav 8 friends","last nav 8 enemies","last nav 9 friends","last nav 9 enemies","last nav 10 friends","last nav 10 enemies","old nav area","old pos x","old pos y","old pos z","old nav 0 friends","old nav 0 enemies","old nav 1 friends","old nav 1 enemies","old nav 2 friends","old nav 2 enemies","old nav 3 friends","old nav 3 enemies","old nav 4 friends","old nav 4 enemies","old nav 5 friends","old nav 5 enemies","old nav 6 friends","old nav 6 enemies","old nav 7 friends","old nav 7 enemies","old nav 8 friends","old nav 8 enemies","old nav 9 friends","old nav 9 enemies","old nav 10 friends","old nav 10 enemies","delta x","delta y","shoot next","crouch next","nav target"};
    vector<long> b = {0,30,0,0,1,6,-896,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,6,-896,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,6,-896,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,6};
    vector<PyObject *> keysToFree, valuesToFree;

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, functionName.c_str());
        /* pFunc is a new reference */

        if (pFunc && PyCallable_Check(pFunc)) {
            int argc = 3;
            pArg = PyDict_New();
            for (int i = 0; i < a.size(); i++) {
                keysToFree.push_back(PyUnicode_FromString(a[i].c_str()));
                valuesToFree.push_back(PyLong_FromLong(b[i]));
                PyDict_SetItem(pArg, keysToFree.back(), valuesToFree.back());
            }
            pValue = PyObject_CallOneArg(pFunc, pArg);
            Py_DECREF(pArg);
            for (int i = 0; i < keysToFree.size(); i++) {
                Py_DecRef(keysToFree[i]);
                Py_DecRef(valuesToFree[i]);
            }
            if (pValue != NULL) {
                printf("Result of call: %ld\n", PyLong_AsLong(pValue));
                Py_DECREF(pValue);
            }
            else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                return 1;
            }
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", functionName.c_str());
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", moduleName.c_str());
        return 1;
    }
    if (Py_FinalizeEx() < 0) {
        return 120;
    }
    return 0;
}