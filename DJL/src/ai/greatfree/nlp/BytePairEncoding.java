package ai.greatfree.nlp;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

import ai.djl.ndarray.NDManager;
import ai.djl.util.Pair;

/**
 * Created: 08/25/2024, Bing Li
 */
final class BytePairEncoding
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			String[] symbols = new String[] { "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "_", "[UNK]" };

			HashMap<String, Integer> rawTokenFreqs = new HashMap<>();
			rawTokenFreqs.put("fast_", 4);
			rawTokenFreqs.put("faster_", 3);
			rawTokenFreqs.put("tall_", 5);
			rawTokenFreqs.put("taller_", 4);

			HashMap<String, Integer> tokenFreqs = new HashMap<>();
			for (Map.Entry<String, Integer> e : rawTokenFreqs.entrySet())
			{
				String token = e.getKey();
				tokenFreqs.put(String.join(" ", token.split("")), rawTokenFreqs.get(token));
			}

			System.out.println(tokenFreqs);

			System.out.println("----------------------");

			int numMerges = 10;
			for (int i = 0; i < numMerges; i++)
			{
				Pair<String, String> maxFreqPair = NLPUtils.getMaxFreqPair(tokenFreqs);
				Pair<HashMap<String, Integer>, String[]> pair = NLPUtils.mergeSymbols(maxFreqPair, tokenFreqs);
				tokenFreqs = pair.getKey();
				symbols = Stream.concat(Arrays.stream(symbols), Arrays.stream(pair.getValue())).toArray(String[]::new);
				System.out.println("merge #" + (i + 1) + ": (" + maxFreqPair.getKey() + ", " + maxFreqPair.getValue() + ")");
			}

			System.out.println("----------------------");
			System.out.println(Arrays.toString(symbols));
			System.out.println("----------------------");
			System.out.println(tokenFreqs.keySet());
			
			System.out.println("----------------------");
			String[] tokens = new String[] {"tallest_", "fatter_"};
			System.out.println(NLPUtils.segmentBPE(tokens, symbols));
		}
	}
}

