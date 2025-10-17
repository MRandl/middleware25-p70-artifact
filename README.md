# Middleware 25 artifact evaluation
You'll need:
- a HuggingFace account with the right to download https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct.
Make sure to agree to the Meta T&C and wait for confirmation (takes a few hours/a day from our experience)
- a machine with more than 64 GB of ram available and an Nvidia GPU that can fit LLama 3.1 8B. Most modern GPUs can do that.

Please clone this repo and run the following:

```
cd docker
docker build -t p70-artifact . \
    --build-arg LDAP_GROUPNAME=SACS-StaffU \
    --build-arg LDAP_GID=11259 \
    --build-arg LDAP_USERNAME=randl \
    --build-arg LDAP_UID=204140

docker run p70-artifact
```

This will create a docker container with all dependencies already included.
