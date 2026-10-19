package ai.greatfree.nlp;

import java.io.IOException;

import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import ai.greatfree.utils.Vocab;

/**
 * Created: 08/24/2024, Bing Li
 */
final class Minibatcher
{
	public static void main(String[] args) throws IOException, TranslateException
	{
		String home = "/Users/libing/Temp/";
		try (NDManager manager = NDManager.newBaseManager())
		{
			Pair<ArrayDataset, Vocab> datasetVocab = NLPUtils.loadDataPTB(home, "ptb/", "ptb.train.txt", 512, 5, 5, manager);
			ArrayDataset dataset = datasetVocab.getKey();
//			Vocab vocab = datasetVocab.getValue();

			Batch batch = dataset.getData(manager).iterator().next();
			String[] names = new String[] { "centers", "contexts_negatives", "masks", "labels" };
			for (int i = 0; i < batch.getData().size(); i++)
			{
				System.out.println(names[i] + " shape: " + batch.getData().get(i).getShape());
			}
		}
	}
}

