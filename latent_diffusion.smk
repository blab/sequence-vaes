configfile: "defaults/config.yaml"

rule all:
    input:
        generated = "results/generated.fasta",
        ordination = "results/ordination.tsv"

rule download_auspice_json:
    output:
        tree = "data/auspice.json",
        root = "data/auspice_root-sequence.json"
    params:
        dataset = config["latent_diffusion"]["dataset"]
    shell:
        """
        nextstrain remote download {params.dataset:q} {output.tree:q}
        """

rule provision_alignment:
    input:
        tree = "data/auspice.json",
        root = "data/auspice_root-sequence.json"
    output:
        alignment = "data/alignment.fasta"
    params:
        gene = config["latent_diffusion"]["gene"]
    shell:
        """
        python3 scripts/alignment.py \
            --tree {input.tree:q} \
            --root {input.root:q} \
            --output {output.alignment:q} \
            --gene {params.gene:q}
        """

rule provision_metadata:
    input:
        tree = "data/auspice.json"
    output:
        metadata = "data/metadata.tsv"
    shell:
        """
        python3 scripts/metadata.py \
            --tree {input.tree:q} \
            --output {output.metadata:q}
        """

rule trim:
    input:
        alignment = "data/alignment.fasta"
    output:
        alignment = "data/trimmed.fasta"
    shell:
        """
        python3 scripts/trim.py \
            --input-alignment {input.alignment:q} \
            --output-alignment {output.alignment:q}
        """

rule train_vae:
    input:
        alignment = "data/trimmed.fasta"
    output:
        vae_model = "models/vae.pth"
    shell:
        """
        python3 latent-diffusion/train_vae.py \
            --input-alignment {input.alignment:q} \
            --output-vae-model {output.vae_model:q}
        """

# rule train:
#     input:
#         alignment = "data/alignment.fasta"
#     output:
#         vae_model = "models/vae.pth",
#         diffusion_model = "models/diffusion.pth"
#     shell:
#         """
#         python latent-diffusion/train.py \
#             --input-alignment {input.alignment:q} \
#             --output-vae-model {output.vae_model:q} \
#             --output-diffusion-model {output.diffusion_model:q}
#         """

# rule generate:
#     input:
#         vae_model = "models/vae.pth",
#         diffusion_model = "models/diffusion.pth"
#     output:
#         alignment = "results/generated.fasta"
#     params:
#         sequence_count = config["latent_diffusion"]["count"]
#     shell:
#         """
#         python latent-diffusion/generate.py \
#             --input-vae-model {input.vae_model:q} \
#             --input-diffusion-model {input.diffusion_model:q} \
#             --output-alignment {output.alignment:q} \
#             --count {params.sequence_count:q}
#         """

rule embed:
    input:
        alignment = "data/trimmed.fasta",
        vae_model = "models/vae.pth"
    output:
        embeddings = "results/embeddings.tsv"
    shell:
        """
        python3 latent-diffusion/embed.py \
            --input-alignment {input.alignment:q} \
            --input-vae-model {input.vae_model:q} \
            --output-embeddings {output.embeddings:q} \
        """

rule compute_ordination:
    input:
        embeddings = "results/embeddings.tsv"
    output:
        ordination = "results/ordination.tsv"
    shell:
        """
        python3 scripts/ordination.py \
            --input {input.embeddings:q} \
            --output {output.ordination:q}
        """
