package ai.greatfree.distributed.cnn.root;

import java.io.IOException;

import org.greatfree.exceptions.RemoteReadException;
import org.greatfree.util.TerminateSignal;

import ai.greatfree.distributed.rnn.RNNConfig;
import edu.greatfree.cluster.ClusterProfile;
import edu.greatfree.cluster.root.UnaryRoot;
import edu.greatfree.exceptions.NoChildrenAvailableException;
import edu.greatfree.exceptions.TaskAlreadyExistedException;

/**
 * Created: 08/28/2024, Bing Li
 */
final class StartRoot
{
	public static void main(String[] args) throws ClassNotFoundException, IOException, TaskAlreadyExistedException, RemoteReadException, NoChildrenAvailableException
	{
		Paths.ROOT().init();
		System.out.println("Root starting up ...");
		UnaryRoot.CLUSTER().start(ClusterProfile.getLightRootSpec(RNNConfig.TRAINING_ROOT, RNNConfig.TRAINING_ROOT_PORT, RNNConfig.REGISTRY_IP, RNNConfig.REGISTRY_PORT), new CNNRootTask());
		System.out.println("Root started ...");
		TerminateSignal.SIGNAL().waitTermination();
	}
}
