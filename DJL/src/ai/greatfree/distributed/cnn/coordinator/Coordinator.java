package ai.greatfree.distributed.cnn.coordinator;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import org.greatfree.exceptions.NullClassConversionException;
import org.greatfree.exceptions.RemoteReadException;
import org.greatfree.framework.container.cs.multinode.message.ShutdownServerNotification;
import org.greatfree.util.IPAddress;
import org.greatfree.util.TerminateSignal;
import org.greatfree.util.Tools;
import org.greatfree.util.UtilConfig;

import ai.djl.util.Pair;
import ai.greatfree.distributed.cnn.CNNConfig;
import ai.greatfree.distributed.cnn.message.CollectGradientsRequest;
import ai.greatfree.distributed.cnn.message.CollectGradientsResponse;
import ai.greatfree.distributed.cnn.message.CNNParam;
import ai.greatfree.distributed.cnn.message.DistributeTrainingDataRequest;
import ai.greatfree.distributed.cnn.message.DistributeTrainingDataResponse;
import ai.greatfree.distributed.cnn.message.InitRequest;
import ai.greatfree.distributed.cnn.message.InitResponse;
import ai.greatfree.distributed.cnn.message.PredictRequest;
import ai.greatfree.distributed.cnn.message.PredictResponse;
import ai.greatfree.distributed.cnn.message.StartTrainingNotification;
import ai.greatfree.distributed.cnn.utils.CNNTools;
import edu.greatfree.cluster.message.ChildrenKeysRequest;
import edu.greatfree.cluster.message.ChildrenKeysResponse;
import edu.greatfree.container.server.PeerContainer;
import edu.greatfree.container.server.PeerSpec;
import edu.greatfree.exceptions.TaskAlreadyExistedException;
import edu.greatfree.framework.cluster.multicast.message.ShutdownChildrenNotification;
import edu.greatfree.framework.cluster.multicast.message.ShutdownRootNotification;

/**
 * CNN counterpart to the RNN Coordinator.
 */
final class Coordinator
{
    private PeerContainer peer;
    private IPAddress rootAddress;
    private Set<String> childrenKeys;
    private AtomicInteger trainingDoneCount;

    public Coordinator()
    {
    }

    private static Coordinator instance = new Coordinator();

    public static Coordinator TRAIN()
    {
        if (instance == null)
        {
            instance = new Coordinator();
            return instance;
        }
        else
        {
            return instance;
        }
    }

    public void stop() throws ClassNotFoundException, IOException, InterruptedException, RemoteReadException
    {
        TerminateSignal.SIGNAL().notifyAllTermination();
        this.peer.stop();
    }

    public void start(PeerSpec spec, String rootName) throws IOException, TaskAlreadyExistedException, ClassNotFoundException, RemoteReadException
    {
        this.peer = new PeerContainer(spec);
        this.peer.start();
        this.rootAddress = this.peer.getPartnerAddressByName(rootName);
        this.trainingDoneCount = new AtomicInteger(UtilConfig.ZERO);
    }

    // ============================================================
    //  GRADIENT COLLECTION  — identical to RNN except for param type
    // ============================================================

    public List<CNNParam> getAllParamsExcept(String childKey)
            throws ClassNotFoundException, RemoteReadException, IOException, NullClassConversionException
    {
        Set<String> childrens = new HashSet<String>(this.childrenKeys);
        childrens.remove(childKey);
        List<CollectGradientsResponse> cgrs = this.peer.read(
                this.rootAddress.getIP(),
                this.rootAddress.getPort(),
                new CollectGradientsRequest(childrens),
                CollectGradientsResponse.class);
        List<CNNParam> params = new ArrayList<CNNParam>();
        for (CollectGradientsResponse entry : cgrs)
        {
            params.addAll(entry.getParams());
        }
        return params;
    }

    public boolean isDone(String childKey)
    {
        if (this.childrenKeys.contains(childKey))
        {
            this.trainingDoneCount.incrementAndGet();
            if (this.trainingDoneCount.get() >= this.childrenKeys.size())
            {
                return true;
            }
        }
        return false;
    }

    public int getDoneCount()
    {
        return this.trainingDoneCount.get();
    }

    // ============================================================
    //  DISTRIBUTE TRAINING DATA  — the shape of this method changes
    // ============================================================

    /**
     * RNN version: ask operator for a text file, read its lines,
     *              send List<String> to each child.
     *
     * CNN version: ask operator for a folder, read all images + labels
     *              + class names, send them to each child.
     */
    public void distributeTrainingData()
            throws IOException, ClassNotFoundException, RemoteReadException, NullClassConversionException
    {
        String folderName;
        List<String> classNames;
        List<byte[]> images;
        List<Integer> labels;
        List<DistributeTrainingDataResponse> dtdrs;

        this.childrenKeys = new HashSet<String>(((ChildrenKeysResponse) this.peer.read(
                this.rootAddress.getIP(),
                this.rootAddress.getPort(),
                new ChildrenKeysRequest())).getChildrenKeys());

        for (String entry : this.childrenKeys)
        {
            System.out.println("Assign image folder for the child, " + entry);
            folderName = Tools.INPUT.nextLine();

            File dir = new File(CNNConfig.SOURCE_DATA_PATH + folderName);
            if (!dir.isDirectory())
            {
                System.out.println("Not a directory: " + dir.getAbsolutePath() + " — skipping.");
                continue;
            }

    
            Pair<Pair<List<String>, List<byte[]>>, List<Integer>> bundle = CNNTools.readImageFolderAsBytes(dir);
            classNames = bundle.getKey().getKey();
            images = bundle.getKey().getValue();
            labels = bundle.getValue();

            System.out.println("Loaded " + images.size() + " images across " + classNames.size() + " classes: " + classNames);

            dtdrs = this.peer.read(
                    this.rootAddress.getIP(),
                    this.rootAddress.getPort(),
                    new DistributeTrainingDataRequest(entry, folderName, classNames, images, labels),
                    DistributeTrainingDataResponse.class);

            for (DistributeTrainingDataResponse res : dtdrs)
            {
                if (!res.isSucceeded())
                    System.out.println("Data is NOT distributed to the child, " + entry + "!");
                else
                    System.out.println("Data is distributed to the child, " + entry + ".");
            }
        }
    }

    // ============================================================
    //  INIT & START TRAINING  — identical to RNN
    // ============================================================

    public void initializeTraining()
            throws ClassNotFoundException, RemoteReadException, IOException, NullClassConversionException
    {
        List<InitResponse> irs = this.peer.read(
                this.rootAddress.getIP(),
                this.rootAddress.getPort(),
                new InitRequest(),
                InitResponse.class);
        for (InitResponse entry : irs)
        {
            if (!entry.isDone())
            {
                System.out.println("Child, " + entry.getChildKey() + ", is NOT initialized successfully!");
            }
            else
            {
                System.out.println("Child, " + entry.getChildKey() + ", is initialized successfully!");
            }
        }
    }

    public void startTraining() throws IOException, InterruptedException
    {
        this.peer.syncNotify(
                this.rootAddress.getIP(),
                this.rootAddress.getPort(),
                new StartTrainingNotification());
    }

    // ============================================================
    //  PREDICT  — operator inputs an image file, not a text prefix
    // ============================================================

    /**
     * RNN version: read a prefix string, print each child's continuation.
     *
     * CNN version: read a path to an image file, send its bytes, print
     *              each child's (label, confidence) prediction.
     */
    public void predict() throws ClassNotFoundException, RemoteReadException, IOException, NullClassConversionException
    {
        System.out.println("Tell me the image file to classify: ");
        String imagePath = Tools.INPUT.nextLine();

        File imageFile = new File(imagePath);
        if (!imageFile.isFile())
        {
            System.out.println("Not a file: " + imageFile.getAbsolutePath());
            return;
        }

        byte[] imageBytes = Files.readAllBytes(imageFile.toPath());
        String fileName   = imageFile.getName();

        List<PredictResponse> prs = this.peer.read(
                this.rootAddress.getIP(),
                this.rootAddress.getPort(),
                new PredictRequest(imageBytes, fileName),
                PredictResponse.class);

        int index = 0;
        for (PredictResponse entry : prs)
        {
            System.out.println(index++ + ") " + entry.getLabel()
                             + "  (" + String.format("%.2f", entry.getConfidence()) + ")");
        }
    }

    // ============================================================
    //  SHUTDOWN  — identical to RNN
    // ============================================================

    public void shutdownChildren() throws IOException, InterruptedException
    {
        this.peer.syncNotify(
                this.rootAddress.getIP(),
                this.rootAddress.getPort(),
                new ShutdownChildrenNotification());
    }

    public void shutdownRoot() throws IOException, InterruptedException
    {
        this.peer.syncNotify(
                this.rootAddress.getIP(),
                this.rootAddress.getPort(),
                new ShutdownRootNotification());
    }

    public void shutdownRegistryServer() throws IOException, InterruptedException
    {
        this.peer.syncNotify(
                CNNConfig.REGISTRY_IP,
                CNNConfig.REGISTRY_PORT,
                new ShutdownServerNotification());
    }
}

