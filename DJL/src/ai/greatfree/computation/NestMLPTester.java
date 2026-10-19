package ai.greatfree.computation;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;

/**
 * Created: 08/19/2024, Bing Li
 */
final class NestMLPTester
{
	public static void main(String[] args) throws TranslateException
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			int inputSize = 20;
			NDArray x = manager.randomUniform(0, 1, new Shape(2, inputSize)); // (2, 20) shape

			SequentialBlock chimera = new SequentialBlock();

			chimera.add(new NestMLP());
			chimera.add(Linear.builder().setUnits(20).build());
			chimera.add(new FixedHiddenMLP());

			chimera.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
			chimera.initialize(manager, DataType.FLOAT32, x.getShape());
			Model model = Model.newInstance("chimera");
			model.setBlock(chimera);

			Translator<NDList, ?> translator = new NoopTranslator();
			NDList xList = new NDList(x);

			Predictor<NDList, ?> predictor = model.newPredictor(translator);
			System.out.println(((NDList) predictor.predict(xList)).singletonOrThrow());
		}
	}
}
