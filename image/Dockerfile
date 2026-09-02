FROM lmsysorg/sglang:deepseek-v4-npu-910b

USER root
COPY debs/ /tmp/dsv4-debs/
RUN dpkg -i /tmp/dsv4-debs/*.deb && \
    ldconfig && \
    rm -rf /tmp/dsv4-debs

WORKDIR /workspace
CMD ["/bin/bash"]
