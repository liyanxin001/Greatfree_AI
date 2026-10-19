package ai.greatfree.rnn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.greatfree.utils.PlotUtils;
import ai.greatfree.utils.TimeMachine;
import ai.greatfree.utils.Vocab;
import tech.tablesaw.plotly.Plot;

/**
 * Created: 08/28/2024, Bing Li
 */
final class NaturalLangStatistics
{
	public static void main(String[] args) throws IOException
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			String[][] tokens = TimeMachine.tokenize(TimeMachine.readTimeMachine(), "word");
			// Since each text line is not necessarily a sentence or a paragraph, we
			// concatenate all text lines
			List<String> corpus = new ArrayList<>();
			for (int i = 0; i < tokens.length; i++)
			{
				for (int j = 0; j < tokens[i].length; j++)
				{
					if (tokens[i][j] != "")
					{
						corpus.add(tokens[i][j]);
					}
				}
			}

			Vocab vocab = new Vocab(new String[][] { corpus.toArray(new String[0]) }, -1, new String[0]);
			for (int i = 0; i < 10; i++)
			{
				Map.Entry<String, Integer> token = vocab.tokenFreqs.get(i);
				System.out.println(token.getKey() + ": " + token.getValue());
			}

			int n = vocab.tokenFreqs.size();
			double[] freqs = new double[n];
			double[] x = new double[n];
			for (int i = 0; i < n; i++)
			{
				freqs[i] = (double) vocab.tokenFreqs.get(i).getValue();
				x[i] = (double) i;
			}

			Plot.show(PlotUtils.plotLogScale(new double[][] { x }, new double[][] { freqs }, new String[] { "" }, "token: x", "frequency: n(x)"));

			System.out.println("--------------------");
			String[] bigramTokens = new String[corpus.size() - 1];
			for (int i = 0; i < bigramTokens.length; i++)
			{
				bigramTokens[i] = corpus.get(i) + " " + corpus.get(i + 1);
			}
			Vocab bigramVocab = new Vocab(new String[][] { bigramTokens }, -1, new String[0]);
			for (int i = 0; i < 10; i++)
			{
				Map.Entry<String, Integer> token = bigramVocab.tokenFreqs.get(i);
				System.out.println(token.getKey() + ": " + token.getValue());
			}

			System.out.println("--------------------");
			String[] trigramTokens = new String[corpus.size() - 2];
			for (int i = 0; i < trigramTokens.length; i++)
			{
				trigramTokens[i] = corpus.get(i) + " " + corpus.get(i + 1) + " " + corpus.get(i + 2);
			}
			Vocab trigramVocab = new Vocab(new String[][] { trigramTokens }, -1, new String[0]);
			for (int i = 0; i < 10; i++)
			{
				Map.Entry<String, Integer> token = trigramVocab.tokenFreqs.get(i);
				System.out.println(token.getKey() + ": " + token.getValue());
			}

			System.out.println("--------------------");
			n = bigramVocab.tokenFreqs.size();
			double[] bigramFreqs = new double[n];
			double[] bigramX = new double[n];
			for (int i = 0; i < n; i++)
			{
				bigramFreqs[i] = (double) bigramVocab.tokenFreqs.get(i).getValue();
				bigramX[i] = (double) i;
			}

			n = trigramVocab.tokenFreqs.size();
			double[] trigramFreqs = new double[n];
			double[] trigramX = new double[n];
			for (int i = 0; i < n; i++)
			{
				trigramFreqs[i] = (double) trigramVocab.tokenFreqs.get(i).getValue();
				trigramX[i] = (double) i;
			}

			Plot.show(PlotUtils.plotLogScale(new double[][] { x, bigramX, trigramX }, new double[][] { freqs, bigramFreqs, trigramFreqs }, new String[] { "unigram", "bigram", "trigram" }, "token: x", "frequency: n(x)"));

			System.out.println("--------------------");
			List<Integer> mySeq = new ArrayList<>();
			for (int i = 0; i < 35; i++)
			{
				mySeq.add(i);
			}

			for (NDList pair : RNNUtils.seqDataIterRandom(mySeq, 2, 5, manager))
			{
				System.out.println("X:\n" + pair.get(0).toDebugString(50, 50, 50, 50, true));
				System.out.println("Y:\n" + pair.get(1).toDebugString(50, 50, 50, 50, true));
			}

			System.out.println("--------------------");
			for (NDList pair : RNNUtils.seqDataIterSequential(mySeq, 2, 5, manager))
			{
				System.out.println("X:\n" + pair.get(0).toDebugString(10, 10, 10, 10, true));
				System.out.println("Y:\n" + pair.get(1).toDebugString(10, 10, 10, 10, true));
			}
		}
	}
}
