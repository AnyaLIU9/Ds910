FROM lmsysorg/sglang:deepseek-v4-npu-910b

USER root
COPY debs/ /tmp/dsv4-debs/
RUN find /tmp/dsv4-debs -maxdepth 1 -type f -name '._*' -delete && \
    dpkg -i /tmp/dsv4-debs/*.deb && \
    ldconfig && \
    rm -rf /tmp/dsv4-debs

WORKDIR /workspace
CMD ["/bin/bash"]
