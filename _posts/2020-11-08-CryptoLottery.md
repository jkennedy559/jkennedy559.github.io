---
layout: post
title:  "CryptoLottery'"
date:   2020-11-08 21:03:36 +0530
---

Having spent the last year trying to get my head around machine learning I starting having a look at Blockchain & Bitcoin. Primarily I want to understand the technology enough to know which companies/use cases are fluff and which may have legs. 

I read through an [IBM developer tutorial](https://developer.ibm.com/technologies/blockchain/tutorials/develop-a-blockchain-application-from-scratch-in-python/) written in Python multiple times and then tried to recreate the implementation from memory. Creating an individual chain and linking the blocks with hash functions isn't difficult to grasp but it's also not what makes Blockchain technology unique. What differentiates the Blockchain from a conventional database is the consensus algorithm and proof of work which allows multiple nodes to reach agreement on what is true. [Here](https://github.com/jkennedy559/Learning/blob/master/blockchain/node.py) is my chain with Flask http endpoints to mine blocks.

I went through some of the [developer build](https://developer.bitcoin.org/devguide/index.html) for the Bitcoin protocol. While the Blockchain is a decentralised database with a consensus mechanism, Bitcoin has features that address problems unique to payments, how units are stored, how to prevent spend without a balance, how to prevent the same balance being spent twice etc etc - and this gets complex quickly. 

I was surprised that the private key for a bitcoin wallet is just a 256-bit random number. I built a [script](https://github.com/jkennedy559/Learning/blob/master/blockchain/lottery.py) to randomly generate a private key and then query the Blockonomics API to check if there is any balance in the address. With a private key identified you could  transfer the BTC elsewhere however the chances of randomly stumbling upon someone else's balance is roughly 10^77. If the chances of winning the daily lottery is 1 in 14 million, your more likely to win the lottery for ten days in a row with a single ticket than find someones coins.





