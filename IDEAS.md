# What I want to implement in dense models

- [ ] Prefill decode disaggregation 

- [ ] Different PP and TP parallelism between prefill and decode (KV cache 
resharding, many to one communication with Send/Recv node) 

- [ ] Support for different hardware (compute power and bandwidth) between prefill and decode

- [ ] Internal roofline model to evaluate what tokens can be batched toghether

- [ ] Support for different communication group on the way we want to arange each node (intra node or inter node KV cache transfer for example)

- [ ] Different way of supporting KV transfer: 
* Bulk transfer with Send/Recv node at the end of prefill stage
* KV cache transfer on the fly, layer by layer, with Send/Recv node at the end of attention computation on each layer (more efficient since it is parallelized with the FFN computation, but more overhead)

- [ ] Different sequence length for input and output tokens

- [ ] Continuous batching with a scheduler (at first only able to determine when different token can be put togheter)

- [ ] Single entry point with specialized parser of the input yaml file

## Possible future features

- [ ] Chunked prefill

- [ ] Scheduler also been able to keep track of current memory occupation and correctly preempt the queue of request of tokens to be generated in the decode step

- [ ] Try to exploit the multi job support of astrasim to simplify the dependencies inn the chakra graph between different requests