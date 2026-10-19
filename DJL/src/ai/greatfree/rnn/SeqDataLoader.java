package ai.greatfree.rnn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.util.Pair;
import ai.greatfree.utils.TimeMachine;
import ai.greatfree.utils.Vocab;

/**
 * Created: 08/28/2024, Bing Li
 */
// public class SeqDataLoader implements Iterable<NDList>
class SeqDataLoader implements Iterable<NDList>
{
	public ArrayList<NDList> dataIter;
	public List<Integer> corpus;
	public Vocab vocab;
	public int batchSize;
	public int numSteps;

	/**
	 * An iterator to load sequence data.
	 */
	public SeqDataLoader(int batchSize, int numSteps, boolean useRandomIter, int maxTokens, NDManager manager) throws IOException, Exception
	{
		Pair<List<Integer>, Vocab> corpusVocabPair = TimeMachine.loadCorpusTimeMachine(maxTokens);
		this.corpus = corpusVocabPair.getKey();
		this.vocab = corpusVocabPair.getValue();

		this.batchSize = batchSize;
		this.numSteps = numSteps;
		if (useRandomIter)
		{
			dataIter = RNNUtils.seqDataIterRandom(corpus, batchSize, numSteps, manager);
		}
		else
		{
			dataIter = RNNUtils.seqDataIterSequential(corpus, batchSize, numSteps, manager);
		}
	}

	@Override
	public Iterator<NDList> iterator()
	{
		/*
		Iterator<NDList> it = dataIter.iterator();
		while (it.hasNext())
		{
//			System.out.println(entry);
			NDList list = it.next();
			for (NDArray dn : list.getResourceNDArrays())
			{
				System.out.println("dn length = " + dn.encode().length);
				System.out.println("dn = " + dn);
			}
		}
		*/
		return dataIter.iterator();
	}
}


