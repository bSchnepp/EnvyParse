# EnvyParse
A parser for large amounts of text, in order
to handle things such as processing an HTML document for keywords,
parsing articles to quickly generate a summary, and for other
similar purposes.

## Why?
I'd like to build something that could help find more relevant
documents around the Internet in an efficient manner.
I *certainly* don't have the hardware resources to go up against
some of those big data people, but I can maybe replicate (on a small scale)
what they do, and maybe do it more efficiently with very limited hardware.

Specifically, it needs to run on a very low power board, and be
very efficient with the little computing power that can be crammed in
~30W.

I'd also *love* to port this all over to HIP/ROCm whenever I can get
the opportunity, so I don't have this weird mount over SSH filesystem
and use git from the local machine on a filesystem on the remote machine
without using something like NFS. It's not very pleasant.
(Though, at that point, 'envy' won't make a lot of sense in the name.)

Currently, I can't do that since the drivers seem a little weird on
my main workstation (where I have a Vega and a Polaris GPU, so no nvcc
anyway).


## Design goals:

	- Parse large documents (greater than 128K) as close to realtime as possible.
	
	- Parse a large number of documents in parallel.
	
	- Obtain information about a document quickly. Use data structures to find and guess possible mistakes, intended words, etc.
	
	- Obtain keywords from a document very quickly.
	
	- Reduce a document down to common points (based on keywords)
	

