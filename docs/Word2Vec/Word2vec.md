# Breaking down word2vec

**Sentences**:

```
if (sentence_length == 0) {
    while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
            real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
            next_random = next_random * (unsigned long long)25214903917 + 11;
            if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
    }
    sentence_position = 0;
}
```







**Learning Rate:**

```c
if (word_count - last_word_count > 10000) {
    word_count_actual += word_count - last_word_count;
    last_word_count = word_count;
    if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
               word_count_actual / (real)(iter * train_words + 1) * 100,
               word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
    }
    alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
    if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
}
```

As with most neural networks, word2vec has a learning rate which changes as training progresses. Although, it would be interesting to see exactly how this learning rate looks when it is graphed over multiple iterations. But first I'll provide some context on what the variables mean:

**`word_count`**: How many words have been processed before this statement has been hit. As we loop over the training algorithm in sentences, we know that `word_count` is some multiple of the variable `MAX_SENTENCE_LENGTH`. This variable is set to zero when a new iteration starts.

**`last_word_count:`** This variable represents the word count at the point of time that the learning rate was last adjusted. This variable is set to zero when a new iteration starts.

**`word_count_actual:`** This is the total amount of words that have been processed within the training. This is **not** set to zero when a new iteration starts - this instead continues to accumulate throughout iterations.

Looking into this segment of code, there are a few initial thoughts - firstly, given the initial `if` statement it's clear that this code within executes once every 10,000 words processed. This is likely because in large training sets, the change of the learning rate is so small that is has an insignificant effect, and therefore there is a minor tradeoff for accuracy for a significant boost in performance. 

Looking into the first two lines after the first `if` statement, we see that the two variables `word_count_actual` and  `last_word_count` are adjusted, so that they correctly reflect he definition I have written about the variables above.
