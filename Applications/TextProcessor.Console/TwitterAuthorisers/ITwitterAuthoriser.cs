using System;
using System.Collections.Generic;
using System.Text;

namespace TwitterProcessor.Console.TwitterAuthorisers
{
    public interface ITwitterAuthoriser
    {
        string GetPinCode(Uri authUrl);
    }
}
