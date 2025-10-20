# Middleware 25 artifact evaluation
You'll need:
- a HuggingFace account with the right to download https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct.
Make sure to agree to the Meta T&C and wait for confirmation (takes a few hours/a day from our experience)
- a machine with more than 64 GB of ram available and an Nvidia GPU that can fit LLama 3.1 8B. Most modern GPUs can do that.

Please clone this repo and run the following:

```
docker build -t p70-artifact . \
    --build-arg LDAP_GROUPNAME=SACS-StaffU \
    --build-arg LDAP_GID=11259 \
    --build-arg LDAP_USERNAME=randl \
    --build-arg LDAP_UID=204140

docker run -d p70-artifact:latest
```

This will create a docker container with all dependencies already included.

You can run code in it with:
```
docker exec -it <<<PRESS TAB>>> bash
```
If the tab press does not automplete for some reason, you can run docker ps and replace the <<<TAB>>> part with the ID of the container you just started.

Once you bashed into it, there are only a few steps left before experimentation. Run:

```
huggingface-cli login
```
and add the HF token of an account that has access to the LLama model we mentioned earlier. When prompted if you want
to add this token to git credentials, say no.




Do not forget to ```docker kill``` your container at the end.
