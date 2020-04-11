function [lwc, wc, wca, ai, result] = Word2vecLearningRateFunction(wc, wca, lwc, g, i, ai, tw)
    global a;
    if (wc - lwc > g)
        wca = wca + wc - lwc;
        lwc = wc;

        a = 0.25 * (1 - wca / (i * tw + 1));
        if a < 0.25 * 0.0001
            a = 0.25 * 0.0001;
        end
    end
    if (wc >= tw)
       ai = ai - 1;
       wc = 0;
       lwc = 0;
    end
    result = a;
end