
MODES = ['aeons', 'naive']
BATCHES = list(range(1, 4))
genome_size = 1  # estimate in Mb

# SHASTA
shasta_config = ["../data/shasta_config/Nanopore-Dec2019_mod.conf"]
# CANU
min_read_len = 10
min_overlap_len = 5
grid = "false"
# REDBEAN
min_read_len = 10


"""
TODO

defined rules for three of the assemblers so far

need some proper data before I can test them anyway


"""



rule all:
    input:
        expand("ShastaRun_{mode}/Assembly.fasta", mode=MODES),
        expand("canu_{mode}/{mode}.seqStore.err", mode=MODES)




rule move_reads:
    """
    After running an aeons experiment, move all resulting sequencing data into a directory
    """
    input:  "seq_data_{batch}_{mode}.fa"
    output: "input_reads/seq_data_{batch}_{mode}.fa"
    shell:  "mv {input} {output}"


rule make_readlist:
    """
    necat needs a list of which files to use, which this creates
    """
    input:  expand("input_reads/seq_data_{batch}_{{mode}}.fa", batch=BATCHES)
    output: "input_reads/reads_{mode}"
    shell:  "find {input} >{output}"


rule concat_reads:
    """
    for most assemblers it's probably easiest if we just concatenate all reads we have
    """
    input:  expand("input_reads/seq_data_{batch}_{{mode}}.fa", batch=BATCHES)
    output: "input_reads/concat_{mode}.fa"
    shell:  "cat {input} >{output}"





rule run_shasta:
    """
    execute the shasta assembler
    - takes arguments in a config file
    - config files available from their github repo
    - for ONT: depends on guppy version used to basecall reads
    - got two generic ones
    - modified the min read length
    
    """
    input: "input_reads/concat_{mode}.fa"
    output: "ShastaRun_{mode}/Assembly.fasta"
    params: conf = shasta_config
    shell:  "shasta --input {input} "
            "--config {params.conf} "
            "--assemblyDirectory ShastaRun_{wildcards.mode}"


rule run_canu:
    """
    execute canu assembly
    - needs minimum 10x coverage (derived from genome estimate)
    - minimum read length and overlap length need to be adjusted to the simulated values
    - useGrid automatically submits an LSF job with seemingly useful defaults
    """
    input: "input_reads/concat_{mode}.fa"
    output: "canu_{mode}/{mode}.seqStore.err"  # TODO this is not the actual output we want
    params:
        genome_size = genome_size,
        min_read_len = min_read_len,
        min_overlap_len = min_overlap_len,
        grid = grid

    shell:  "canu "
            "-p {wildcards.mode} "
            "-d canu_{wildcards.mode} "
            "-genomeSize={params.genome_size}m "
            "-minReadLength={params.min_read_len} "
            "-useGrid={params.grid} "
            "-minOverlapLength={params.min_overlap_len} "
            "-nanopore {input}"


rule run_redbean_assembler:
    """
    execute the readbean assembly
    -x preset   ont / preset2 for <1G
    -g might need genome estimate
    -X choose best X depth from input (attention: default is 50) - only active with -g
    -L min read length
    -l min alignment length
    -m min matched length by kmer matching
    """
    input: "input_reads/concat_{mode}.fa"
    output: "redbean_out_{mode}/redbean_{mode}.ctg.lay.gz"
    params:
        # genome_size = genome_size,
        min_read_len = min_read_len

    shell:
        """
        wtdbg2 -x preset2 -L {params.min_read_len} -o redbean_{wildcards.mode} -i {input}
        mkdir redbean_out_{wildcards.mode}
        mv redbean_{wildcards.mode} redbean_out_{wildcards.mode} 
        """


rule run_redbean_consenser:
    """
    run the redbean consenser (2nd step)
    """
    input: "redbean_out_{mode}/redbean_{mode}.ctg.lay.gz"
    output: "redbean_out_{mode}/redbean_{mode}.ctg.fa"

    shell:
        "wtpoa-cns -i {input} -fo {output}"