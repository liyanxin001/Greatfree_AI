package ai.greatfree.linearregression;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import tech.tablesaw.api.FloatColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api.ScatterPlot;

/**
 * Created: 08/13/2024, Bing Li
 */
public final class SyntheticData
{
	public static DataPoints create(NDManager manager, NDArray w, float b, int numExamples)
	{
		NDArray X = manager.randomNormal(new Shape(numExamples, w.size()));
		NDArray y = X.dot(w).add(b);
		// Add noise
		y = y.add(manager.randomNormal(0, 0.01f, y.getShape(), DataType.FLOAT32));
		return new DataPoints(X, y);
	}

	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray trueW = manager.create(new float[] { 2, -3.4f });
			float trueB = 4.2f;

			DataPoints dp = create(manager, trueW, trueB, 1000);
			NDArray features = dp.getX();
			NDArray labels = dp.getY();

			System.out.printf("features: [%f, %f]\n", features.get(0).getFloat(0), features.get(0).getFloat(1));
			System.out.println("label: " + labels.getFloat(0));

			float[] X = features.get(new NDIndex(":, 1")).toFloatArray();
			float[] y = labels.toFloatArray();

			Table data = Table.create("Data").addColumns(FloatColumn.create("X", X), FloatColumn.create("y", y));
			Plot.show(ScatterPlot.create("Synthetic Data", data, "X", "y"));
		}
	}
}
