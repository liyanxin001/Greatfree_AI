package ai.greatfree.utils;

import ai.greatfree.linearregression.SoftMaxProfile;
import tech.tablesaw.api.ColumnType;
import tech.tablesaw.api.FloatColumn;
import tech.tablesaw.api.Row;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.columns.Column;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api.LinePlot;
import tech.tablesaw.plotly.components.Figure;

/**
 * Created: 08/14/2024, Bing Li
 */
public class Animator
{
	private String id; // Id reference of graph(for updating graph)
	private Table data; // Data Points

	public Animator()
	{
		this.id = "";

		// Incrementally plot data
		this.data = Table.create("Data").addColumns(FloatColumn.create("epoch", new float[] {}), FloatColumn.create("value", new float[] {}), StringColumn.create("metric", new String[] {}));
	}

	// Add a single metric to the table
	public void add(float epoch, float value, String metric)
	{
		Row newRow = this.data.appendRow();
		newRow.setFloat("epoch", epoch);
		newRow.setFloat("value", value);
		newRow.setString("metric", metric);
	}

	// Add accuracy, train accuracy, and train loss metrics for a given epoch
	// Then plot it on the graph
	public void add(float epoch, float accuracy, float trainAcc, float trainLoss)
	{
		add(epoch, trainLoss, "train loss");
		add(epoch, trainAcc, "train accuracy");
		add(epoch, accuracy, "test accuracy");
		show();
	}

	// Display the graph
	public void show()
	{
		if (this.id.equals(""))
		{
			// id = display(LinePlot.create("", data, "epoch", "value", "metric"));
//			Plot.show(LinePlot.create("", data, "epoch", "value", "metric"));
			this.id = display(LinePlot.create("", data, "epoch", "value", "metric"));
			return;
		}
		update();
	}

	// Update the graph
	public void update()
	{
		updateDisplay(this.id, LinePlot.create("", data, "epoch", "value", "metric"));
	}

	/*
	 * The method is created by me. I am not sure it is correct or not! 08/15/2024, Bing Li
	 */
	public void updateDisplay(String id, Figure figure)
	{
		Plot.show(figure, id);
//		Plot.show(figure);
	}
	
	/*
	 * The method is created by me. I am not sure it is correct or not! 08/15/2024, Bing Li
	 */
	public String display(Figure figure)
	{
		Plot.show(figure);
		return SoftMaxProfile.TRAIN().getFigurePrefix() + figure.hashCode();
	}

	// Returns the column at the given index if it is a float column
	// Otherwise returns null
	public float[] getY(Table t, int index)
	{
		Column<?> c = t.column(index);
		if (c.type() == ColumnType.FLOAT)
		{
			float[] newArray = new float[c.size()];
			System.arraycopy(c.asList().toArray(), 0, newArray, 0, c.size());
			return newArray;
		}
		return null;
	}
}
