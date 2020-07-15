g++ -shared -fPIC -std=c++11 mypass_lib.cc -o libmypass_lib.so -I /shangz-dt/incubator-mxnet/include/mxnet
g++ -shared -fPIC -std=c++11 pass_fuseFC.cc -o passfuseFC_lib.so -I /shangz-dt/incubator-mxnet/include/mxnet
cp libmypass_lib.so ../
cp passfuseFC_lib.so ../
