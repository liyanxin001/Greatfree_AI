package ai.greatfree.computation;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;

/**
 * Created: 08/19/2024, Bing Li
 */
final class CustomBlock
{
	public static void main(String[] args) throws TranslateException
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
//			int inputSize = 20;
			int inputSize = 1;
			NDArray x = manager.randomUniform(0, 1, new Shape(2, inputSize)); // (1, 20) shape
			Model model = Model.newInstance("lin-reg");

			MLP net = new MLP(inputSize);
			net.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
			net.initialize(manager, DataType.FLOAT32, x.getShape());

			model.setBlock(net);

			Translator<NDList, ?> translator = new NoopTranslator();
			NDList xList = new NDList(x);

			Predictor<NDList, ?> predictor = model.newPredictor(translator);
			System.out.println(((NDList) predictor.predict(xList)).singletonOrThrow());
		}
	}
}

