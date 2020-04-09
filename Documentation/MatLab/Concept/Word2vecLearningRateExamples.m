granularity = 1000;
train_words = 10000;
iterations = 5;
actual_iterations = 5;
word_count = 0;
word_count_actual = 0;
last_word_count = 0;

results = [];
while actual_iterations > 0
    word_count = word_count + 1;
    [last_word_count, word_count, word_count_actual, actual_iterations, result] = Word2vecLearningRateFunction(word_count, word_count_actual, last_word_count, granularity, iterations, actual_iterations, train_words);
    results = [results, result];
end

h = plot(1:train_words * iterations, results, '.');
ylim([0 0.25])