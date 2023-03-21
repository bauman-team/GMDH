FROM quay.io/pypa/manylinux1_x86_64:latest

RUN yum install -y epel-release gcc gcc-c++ gcc-gnat libgcc libgcc.i686 glibc-devel bison flex texinfo build-essential wget xz

RUN yum groupinstall -y 'Development Tools' || exit 0

RUN wget http://ftp.gnu.org/gnu/gcc/gcc-8.2.0/gcc-8.2.0.tar.xz && unxz gcc-8.2.0.tar.xz && tar -xf gcc-8.2.0.tar && \
cd gcc-8.2.0 && ./contrib/download_prerequisites && mkdir build && cd build && ../configure --enable-languages=c,c++ --disable-multilib && \ 
make -j4 && make install && export PATH="/usr/local/bin:$PATH" && export CC=/usr/local/bin/gcc CXX=/usr/local/bin/g++

RUN echo "export CC=/usr/local/bin/gcc CXX=/usr/local/bin/g++" >> /etc/environment && echo "export PATH=/usr/local/bin:$PATH" >> /etc/bashrc

RUN yum install -y libicu libicu-devel

RUN git clone http://github.com/boostorg/boost.git && cd boost && git checkout boost-1.79.0 && git submodule update --init

RUN export PATH="/usr/local/bin:$PATH" && export CC=/usr/local/bin/gcc CXX=/usr/local/bin/g++ && cd boost && ./bootstrap.sh --with-libraries=filesystem,system,thread,json,chrono --prefix=/usr/local && \
./tools/build/src/engine/bjam install --prefix=/usr/local --with-filesystem --with-system --with-thread --with-json --with-chrono cxxflags="-fPIC" link=static runtime-link=shared threading=multi variant=release

RUN /opt/python/cp37-cp37m/bin/python -m pip install cmake==3.20.2 && ln -s /opt/python/cp37-cp37m/bin/cmake /usr/bin/cmake

RUN yum -y install openssl-devel libffi-devel bzip2-devel

RUN git clone https://github.com/python/cpython.git

# python3.10
RUN export PATH="/usr/local/bin:$PATH" && export CC=/usr/local/bin/gcc CXX=/usr/local/bin/g++ && cd cpython && git checkout v3.10.10 && \
echo "$(sed 's/O_CLOEXEC/FD_CLOEXEC/' Python/perf_trampoline.c)" > Python/perf_trampoline.c && \
echo "$(sed 's/-Werror=implicit-function-declaration//' configure)" > configure && ./configure --enable-optimizations --prefix=/opt/python/cp310-cp310m/ && \
make && make install

RUN rm -rf cpython

RUN git clone https://github.com/python/cpython.git

# python3.11
RUN export PATH="/usr/local/bin:$PATH" && export CC=/usr/local/bin/gcc CXX=/usr/local/bin/g++ && cd cpython && git checkout v3.11.2 && \
echo "$(sed 's/O_CLOEXEC/FD_CLOEXEC/' Python/perf_trampoline.c)" > Python/perf_trampoline.c && \
echo "$(sed 's/-Werror=implicit-function-declaration//' configure)" > configure && ./configure --enable-optimizations --prefix=/opt/python/cp311-cp311m/ && \
make && make install

COPY wheel-0.40.0-py3-none-any.whl .

COPY ./compile_whl_dist.sh .

CMD ["/bin/bash"]