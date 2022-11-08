FROM 476020832721.dkr.ecr.ca-central-1.amazonaws.com/qunex_baseline:1.0.0

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_SESSION_TOKEN
RUN echo $AWS_ACCESS_KEY_ID
RUN echo $AWS_SECRET_ACCESS_KEY
RUN echo $AWS_SESSION_TOKEN


RUN source /opt/qunex/env/qunex_environment.sh && \
   pip3 install awscli && \
    pip3 install watchdog pandas numpy boto3 pydicom && \
    aws configure set aws_access_key_id  $AWS_ACCESS_KEY_ID && \
    aws configure set aws_secret_access_key  $AWS_SECRET_ACCESS_KEY && \
    aws configure set aws_session_token  $AWS_SESSION_TOKEN && \
    aws configure set region us-east-1  && \
    mkdir /imaging-features  && \
    aws s3 cp s3://obi-datalake/research/imaging/datasets/CANBIND/reference/ /imaging-features/utils/reference/ --recursive --region=ca-central-1

##
# Dockerfile for QuNex suite
##
# Install the PIP Python package manager

COPY qunex-scripts/gp_HCP.py /opt/qunex/niutilities/niutilities/HCP/gp_HCP.py
COPY qunex-scripts/process_hcp.py /opt/qunex/python/qx_utilities/hcp/process_hcp.py

# Install miniconda comes with conda, python3.x and pip 21.x
RUN yum -y install curl bzip2 \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local/ \
    && rm -rf /tmp/miniconda.sh \
    && rpm -e --nodeps curl bzip2

RUN yum install -y epel-release

RUN yum install -y tcsh libXp openmotif gsl xorg-x11-fonts-misc        \
                      PyQt4 R-devel netpbm-progs gnome-tweak-tool ed    \
                      libpng12 xorg-x11-server-Xvfb firefox             \
                      python3-matplotlib
RUN yum -y install curl
WORKDIR /
RUN curl -O https://afni.nimh.nih.gov/pub/dist/bin/misc/@update.afni.binaries
RUN tcsh @update.afni.binaries -package linux_centos_7_64 -do_extras
RUN export R_LIBS=$HOME/R && echo "$R_LIBS" && mkdir $R_LIBS
RUN echo  'setenv R_LIBS ~/R'     >> ~/.cshrc
RUN echo  'export R_LIBS=$HOME/R' >> ~/.bashrc
RUN . ~/.bashrc
RUN $HOME/abin/rPkgsInstall -pkgs ALL

WORKDIR /
RUN yum -y install wget

# Version update to latest
RUN conda update -n base -c defaults conda

# Create and activate virtual environment to install imaging-features dependencies
RUN conda create -n imaging_features \
    && source activate imaging_features

# Copy imaging-features pipeline base code
COPY . /imaging-features/
WORKDIR /imaging-features

#Install package dependencies
RUN yum install -y libjpeg-devel
RUN yum install -y zlib-devel
RUN yum install -y libxml2-devel
RUN  yum -y install freetype-devel
RUN yum install -y  python3-devel
RUN pip3 install -r requirements.txt
RUN pip3 install --user quilt3 
#RUN pip3 install --user git+https://github.com/Deep-MI/qatools-python.git@freesurfer-module-releases#egg=qatoolspython
#RUN pip3 install --user --src /imaging-features/utils/ --editable git+https://github.com/Deep-MI/qatools-python.git@freesurfer-module-releases#egg=qatoolspython
#Install pipeline reference libraries from Quilt
#setup R configs
RUN Rscript -e "install.packages('optparse')"
RUN Rscript -e "install.packages('docstring')"
RUN Rscript -e "install.packages('igraph')"

ENTRYPOINT ["bash"]
