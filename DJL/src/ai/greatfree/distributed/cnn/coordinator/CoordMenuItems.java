package ai.greatfree.distributed.cnn.coordinator;

/**
 * Created: 09/01/2024, Bing Li
 */
public final class CoordMenuItems
{
	public final static String TAB = "\t";
	public final static String MENU_HEAD = "\n========== Menu Head ===========";

	public final static String DISTRIBUTE_TRAINING_DATA = TAB + "1) Distribute Training Data";
	public final static String INITIALIZAE = TAB + "2) Initialize";
	public final static String START_TRAINING = TAB + "3) Start Training";
	public final static String PREDICT = TAB + "4) Predict";
	public final static String SHUTDOWN_CHILDREN = TAB + "5) Shutdown Children";
	public final static String SHUTDOWN_ROOT = TAB + "6) Shutdown Root";
	public final static String SHUTDOWN_REGISTRY_SERVER = TAB + "7) Shutdown Registry Server";

	public final static String QUIT = TAB + "0) Quit";
	public final static String MENU_TAIL = "========== Menu Tail ===========\n";
	public final static String INPUT_PROMPT = "Input an option:";
	public final static String WRONG_OPTION = "Wrong option!";
}
