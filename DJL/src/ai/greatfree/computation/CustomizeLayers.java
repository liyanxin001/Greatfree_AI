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

/**
 * Created: 08/19/2024, Bing Li
 */
final class CustomizeLayers
{
	public static void main(String[] args) throws TranslateException
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			CenteredLayer layer = new CenteredLayer();

			Model model = Model.newInstance("centered-layer");
			model.setBlock(layer);

			Predictor<NDList, NDList> predictor = model.newPredictor(new NoopTranslator());
			NDArray input = manager.create(new float[] { 1f, 2f, 3f, 4f, 5f });
			System.out.println(((NDList) predictor.predict(new NDList(input))).singletonOrThrow());

			System.out.println("------------------------");

			SequentialBlock net = new SequentialBlock();
			net.add(Linear.builder().setUnits(128).build());
			net.add(new CenteredLayer());
			net.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
			net.initialize(manager, DataType.FLOAT32, input.getShape());

			input = manager.randomUniform(-0.07f, 0.07f, new Shape(4, 8));
			NDArray y = predictor.predict(new NDList(input)).singletonOrThrow();
			System.out.println(y.mean());
		}
	}
}

