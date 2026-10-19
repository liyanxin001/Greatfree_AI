package ai.greatfree.preliminaries;

import java.io.IOException;
import java.util.List;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import tech.tablesaw.api.BooleanColumn;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.columns.Column;

/**
 * Created: 08/09/2024, Bing Li
 */
final class HandleMissingData
{
	@SuppressWarnings("unchecked")
	public static void main(String[] args) throws IOException
	{
		String file = "/Users/libing/Temp/house_tiny.csv";
		Table data = Table.read().file(file);
		System.out.println(data);
		
		System.out.println("===================================================");

		@SuppressWarnings("static-access")
		Table inputs = data.create(data.columns());
		// Table inputs = Table.create(data.columns());
		inputs.removeColumns("Price");

		@SuppressWarnings("deprecation")
		Table outputs = data.select("Price");

		@SuppressWarnings("rawtypes")
		Column col = inputs.column("NumRooms");
		col.set(col.isMissing(), (int) inputs.nCol("NumRooms").mean());
		System.out.println(inputs);

		System.out.println("===================================================");

		StringColumn strCol = (StringColumn) inputs.column("Alley");
		List<BooleanColumn> dummies = strCol.getDummies();
		inputs.removeColumns(strCol);
		inputs.addColumns(DoubleColumn.create("Alley_Pave", dummies.get(0).asDoubleArray()), DoubleColumn.create("Alley_nan", dummies.get(1).asDoubleArray()));
		System.out.println(inputs);
		
		NDManager nd = NDManager.newBaseManager();
		NDArray x = nd.create(inputs.as().doubleMatrix());
		NDArray y = nd.create(outputs.as().intMatrix());

		System.out.println("===================================================");
		System.out.println(x);
		System.out.println("===================================================");
		System.out.println(y);
	}
}

