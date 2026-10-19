package ai.greatfree.rnn;

import java.util.Arrays;
import java.util.List;

import ai.djl.util.Pair;
import ai.greatfree.utils.Vocab;

/**
 * Created: 08/28/2024, Bing Li
 */
final class TextPreprocess
{
	public static void main(String[] args) throws Exception
	{
		String[] lines = RNNUtils.readTimeMachine();
		System.out.println("# text lines: " + lines.length);
		System.out.println(lines[0]);
		System.out.println(lines[10]);

		System.out.println("----------------------");

		String[][] tokens = RNNUtils.tokenize(lines, "word");
		for (int i = 0; i < 11; i++)
		{
			System.out.println(Arrays.toString(tokens[i]));
		}

		/*
		 * System.out.println("----------------------");
		 * 
		 * tokens = RNNUtils.tokenize(lines, "char"); for (int i = 0; i < 11; i++) {
		 * System.out.println(Arrays.toString(tokens[i])); }
		 */

		System.out.println("----------------------");

		Vocab vocab = new Vocab(tokens, 0, new String[0]);
		for (int i = 0; i < 10; i++)
		{
			String token = vocab.idxToToken.get(i);
			System.out.print("(" + token + ", " + vocab.tokenToIdx.get(token) + ") ");
		}

		System.out.println("\n----------------------");

		for (int i : new int[] { 0, 10 })
		{
			System.out.println("Words:" + Arrays.toString(tokens[i]));
			System.out.println("Indices:" + Arrays.toString(vocab.getIdxs(tokens[i])));
		}

		System.out.println("\n----------------------");

		Pair<List<Integer>, Vocab> corpusVocabPair = RNNUtils.loadCorpusTimeMachine(-1);
		List<Integer> corpus = corpusVocabPair.getKey();
		vocab = corpusVocabPair.getValue();

		System.out.println(corpus.size());
		System.out.println(vocab.length());
	}
}

