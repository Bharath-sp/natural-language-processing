---
layout: post
title: Discourse
categories: [NLP]
toc: true
---

* TOC
{:toc}

## Introduction
Languages normally consists of collocated, structured, coherent groups of sentences. We refer to such a coherent structured group of sentences as a **discourse**, and we use the word **coherence** to refer to the relationship between sentences that makes real discourses different from just random assemblages of sentences. Coherence provides meaningful connection between sentences.

What makes a discourse coherent? Real discourses exhibit both local coherence and global coherence.

## Local Coherence
Let's consider three ways in which real discourses are locally coherent.

1. Sentences or clauses in real discourses are related to nearby sentences in systematic ways.

* John took a train from Paris to Istanbul. He likes spinach.
* Jane took a train from Paris to Istanbul. She had to attend a conference.

The second pair gives a REASON for Jane's action. Structured relationships like REASON that hold between text units are called coherence relations, and coherent discourses are structured by many such coherence relations. Other relations are elaboration, evidence, attribution, list, etc.

2. A second way a discourse can be locally coherent is by virtue of being "about" someone or something. In sentences, some entities are salient, and the discourse focuses on them and doesn't go back and forth between multiple entities. This is called **entity-based coherence**.

* John wanted to buy a piano for his living room.
* Jenny also wanted to buy a piano.
* He went to the piano store.
* It was nearby.
* The living room was on the second floor.
* She didn’t find anything she liked.
* The piano he bought was hard to get up to that floor.

This is an incoherent passage, in which the salient entity seems to wildly swing from John to Jenny to the piano store to the living room, back to Jenny, then the piano again.

3. Finally, discourses can be locally coherent by being topically coherent: nearby sentences are generally about the same topic and use the same or similar vocabulary to discuss these topics. For example, the fact that the words house, chimney, garret, closet, and window - all of which belong to the same semantic field is a cue that the sentences are tied together as a discourse.

Coherence is a property of a well-written text. Coherence detection helps in measuring the quality of a text.

## Miscellaneous
* **Lexical chains**: All words that refer to the same topic. Different from the coreference chain.
* **Entity grid**: Rows are entities and columns are sentences. For each entity, we see its transition across sentences. In a real discourse, the transitions are mostly smooth such as {S, X, -, -} and {X, -, -}. And not abrupt such as {X, -, S}. This transition property can help in measuring/detecting coherence in an automated manner. It can be used to check if the given text is a real discourse or not