---
layout: post
title: WordPiece Tokenization
categories: [NLP]
toc: true
---

* TOC
{:toc}


## Introduction
WordPiece is a subword tokenization algorithm closely related to Byte Pair Encoding (BPE). Developed by Google, it is used for pre-training the models like BERT, DistilBERT, and other transformer-based architectures. WordPiece and BPE share the same fundamental concept: iteratively merging frequent pairs of characters or subwords to create a vocabulary of subword units. However, WordPiece introduces a key modification in how it selects pairs to merge.

## WordPiece Algorithm

### Training

Step 1: **Initialization:**

1. Start with a vocabulary of individual characters from the training corpus.
2. Add special tokens: [UNK] and `##` prefix for subwords that don't start a word.

For example: Say our corpus is "the quick brown fox jumps over the lazy dog". Then, the base vocabulary is:

[`UNK`, `t`, `##h`, `##e`, ` `, `q`, `##u`, `##i`, `##c`, `##k`, `b`, `##r`, `##o`, `##w`, `##n`, `f`, `##x`, `j`, `##m`, `##p`, `##s`, `o`, `##v`, `l`, `##a`, `##z`, `##y`, `d`, `##g`]

Initial tokenization of our corpus:

[`t`, `##h`, `##e`, ` `, `q`, `##u`, `##i`, `##c`, `##k`, ` `, `b`, `##r`, `##o`, `##w`, `##n`, ` `, `f`, `##o`, `##x`, ` `, `j`, `##u`, `##m`, `##p`, `##s`, ` `, `o`, `##v`, `##e`, `##r`, ` `, `t`, `##h`, `##e`, ` `, `l`, `##a`, `##z`, `##y`, ` `, `d`, `##o`, `##g`]

Step 2: **Calculating the Score:**

For each pair of adjacent tokens, assign a likelihood-based score:

$$
\text{Score}(A, B) = \frac{\text{freq}(AB)}{\text{freq(A)} * \text{freq(B)}}
$$

The score effectively measures how much more likely the pair is to appear together than would be expected if they were independent. Unlike BPE, which merges the most frequent pairs, WordPiece prioritizes merges that maximize the likelihood of the training data.

For example,

* For the pair, (`t`, `##h`): Score = $\frac{2}{2 * 2} = 0.5$.
* (`q`, `##u`): Score = $\frac{1}{1 * 1} = 1$.

By dividing by the product of individual frequencies, the algorithm penalizes pairs where the individual parts are already very common (like `th` and `e`), preventing them from being merged just because they are frequent.

Step 3: **Merging**

1. Select the pair with the highest score, say we selected `q` and `##u`.
<br>
2. Merge them, and add the merged token to the vocabulary. The updated vocabulary is:
[`UNK`, `t`, `##h`, `##e`, ` `, `q`, `##u`, `##i`, `##c`, `##k`, `b`, `##r`, `##o`, `##w`, `##n`, `f`, `##x`, `j`, `##m`, `##p`, `##s`, `o`, `##v`, `l`, `##a`, `##z`, `##y`, `d`, `##g`, `qu`]

If it's not a word-initial subword, add it with the `##` prefix. For example, if we are merging (`##g`, `##s`), the merged token will be `##gs`.

<br>
3. Update the tokenization of the training corpus to use this new token where applicable. Updated tokenization of the training corpus:

[`t`, `##h`, `##e`, ` `, `qu`, `##i`, `##c`, `##k`, ` `, `b`, `##r`, `##o`, `##w`, `##n`, ` `, `f`, `##o`, `##x`, ` `, `j`, `##u`, `##m`, `##p`, `##s`, ` `, `o`, `##v`, `##e`, `##r`, ` `, `t`, `##h`, `##e`, ` `, `l`, `##a`, `##z`, `##y`, ` `, `d`, `##o`, `##g`]

Step 4: **Stopping Criteria**

Repeat step 2 and 3 until the desired vocabulary size is reached or until the highest merge score falls below a certain threshold. A score threshold can help ensure that only meaningful merges are performed.

As the algorithm progresses, it will start to form complete words and common subwords: `quick`, `brown`, `jumps`, `over`, `lazy`, and common prefixes and suffixes like `##ing`, `##ed`, `##s`.

The algorithm's focus on likelihood improvement rather than simple frequency often results in more linguistically meaningful subword units.

**Final Vocabulary:**
A subset of the final vocabulary might look like: [`UNK`, `the`, `quick`, `brown`, `fox`, `jumps`, `over`, `lazy`, `dog`, `##s`, `##ing`, `##ed`, `qu`, `##i`, `##ck`, `br`, `##own`, `ju`, `##mp`, `la`, `##zy`, ...]

**Tokenization with Final Vocabulary**:
Using this vocabulary, our training corpus would be tokenized as: [`the`, `quick`, `brown`, `fox`, `jumps`, `over`, `the`, `lazy`, `dog`].

**Key Observations**:
1. Common words (`the`, `quick`, `brown`) become single tokens.
2. Less common words may be split (e.g., `jumps` might be `jump` + `##s` in a larger corpus). It balances the use of larger subword units where ever possible with the ability to fall back to smaller units when needed.
3.  The algorithm balances frequency with the improvement in likelihood, leading to linguistically sensible subwords.
4. The `##` prefix helps maintain information about word boundaries and morphology.

### WordPiece Encoder
Once the WordPiece vocabulary is constructed, tokenization of new text follows these steps. Note here there is no learned merge rules here, we only save the final vocabulary set from training.

We break any input word into a sequence of known subword units from the vocabulary. WordPiece uses a greedy longest-match-first strategy.

1. Split the input text into words by whitespace character.
2. For each word:
a. Try to match the longest subword starting from the current position that exists in the vocabulary (from left to right).
b. If a match is found, add it to the output and repeat from the next character.
c. If no match is found, assign `UNK` to the **whole** word.

For matches after the first character, use the ``##`` version of the subword.

**Example:**

* Sentence: "The quickness of the brown fox surprised the lazy dog".
Tokenization: [`the`, `qu`, `##ick`, `##ness`, `of`, `the`, `brown`, `fox`, `UNK`, `the`, `lazy`, `dog`]

For the word `quickness`:

* Longest match from start: `quickness` ❌ (not in vocab)
* `quicknes` ❌
* `quickne` ❌
* `quickn` ❌
* `quick` ❌
* `quic` ❌
* `qui` ❌
* `qu` ✔️

So, the first token is `qu`, and the remaining is `##ickness`. Now match with continuation tokens:

* `##ickness` ❌
* `##icknes` ❌
* `##ickne` ❌
* `##ickn` ❌
* `##ick` ✔️

The tokens are `qu`, `##ick`, and the remaining is `##ness`.

* `##ness` ✔️

Thus, the final tokens are `quickness` $\to$ [`qu`, `##ick`, `##ness`].

If at any step the remaining substring cannot be fully matched, the entire word is replaced with `UNK`.

For the word `surprised`:

* `surprised` ❌
* `surprise` ❌
* `surpris` ❌
* `surpri` ❌
* `surpr` ❌
* `surp` ❌
* `sur` ❌
* `su` ❌
* `s` ✔️

So, the first token is `s`, and the remaining is `##urprised`. Then, we repeat this with the remaining substring. Suppose `##u` is not found in the vocab, then we discard everything and return `UNK`.

WordPiece enforces that the entire word must be decomposable into valid tokens. Otherwise, discard everything and return `UNK`. This is a deliberate modeling choice to prevent mixing valid token `s` with meaningless fragments `UNK`.

<div class="admonition tip">
  <p class="admonition-title">TIP</p>
  <p>When trained on a large corpus, the algorithm rarely produce `UNK` in practice despite this strict rule.</p>
</div>

The tokenization process uses a greedy approach, matching the longest possible subword from left to right. This way the WordPiece balances the use of whole words, subwords, and the `UNK` token to handle both seen and unseen words.