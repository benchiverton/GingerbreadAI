using System;
using System.Collections.Generic;
using System.Text;

namespace TwitterProcessor.Console.Observers
{
    public interface ITweetObserver<in TObservable, out TOut> : IObserver<TObservable>
    {
        event Action<TOut> ProcessTweet;
        event Action StartNewObserver;
    }
}
