using UnityEngine;
using Unity.Sentis;
using System;
using System.Globalization;
using Unity.Mathematics;
using UnityEngine.UI;

public class BlazeFaceRunner : MonoBehaviour
{
    [SerializeField]private ModelAsset BlazeFaceAsset;
    private Model BlazeFaceModel;
    private Worker worker;
    private Tensor<float> inputTensor, outputTensor;
    private TextureTransform transform;
    [SerializeField] private TextAsset anchorCSV;
    const int k_NumAnchors = 896;
    float[,] m_Anchors;

    public float iouThreshold = 0.3f;
    public float scoreThreshold = 0.5f;

    [SerializeField] private Texture2D tex;
    public RawImage rawImage;
    private RenderTexture renderTexture;

    public static float[,] LoadAnchors(string csv, int k)
    {
        var anchors = new float[k, 4];
        var anchorLines = csv.Split('\n');

        for (int i = 0; i < k; i++)
        {
            var anchorValues = anchorLines[i].Split(',');
            for (var j = 0; j < 4;  j++)
            {
                anchors[i, j] = float.Parse(anchorValues[j], CultureInfo.InvariantCulture);
            }
        }
        return anchors;
    }

    //Non-Maximum Suppression 
    public static (FunctionalTensor, FunctionalTensor, FunctionalTensor) NWSF(FunctionalTensor rawBoxes, FunctionalTensor rawScores, FunctionalTensor anchors, float iouTreshold, float scoreThreshold)
    {
        //uv
        var xCenter = rawBoxes[0, .., 0] + anchors[.., 0];
        var yCenter = rawBoxes[0, .., 1] + anchors[.., 1] * 128;  

        var widthHalf = 0.5f * rawBoxes[0, .., 2];
        var heightHalf = 0.5f * rawBoxes[0, .., 3];

        var nmsBoxes = Functional.Stack(new[]
        {
            yCenter - heightHalf,
            xCenter - widthHalf,
            yCenter + heightHalf,
            xCenter + widthHalf,
        }, 1);

        var clamp = Functional.Sigmoid(Functional.Clamp(rawScores, -100f, 100f));
        var nmsScores = Functional.Squeeze(clamp);
        var selectedIndies = Functional.NMS(nmsBoxes, nmsScores, iouTreshold, scoreThreshold);

        var selectedBoxes = Functional.IndexSelect(rawBoxes, 1, selectedIndies).Unsqueeze(0);
        var selectedScores = Functional.IndexSelect(rawScores, 1, selectedIndies).Unsqueeze(0);

        return (selectedIndies, selectedScores, selectedBoxes);

    }

    void Start()
    {
        m_Anchors = LoadAnchors(anchorCSV.text, k_NumAnchors);

        BlazeFaceModel = ModelLoader.Load(BlazeFaceAsset);
        var graph = new FunctionalGraph();

        var input = graph.AddInput(BlazeFaceModel, 0);
        var outputs = Functional.Forward(BlazeFaceModel,　2 * input - 1);

        var boxes = outputs[0];
        var scores = outputs[1];
        var ancohorData = new float[k_NumAnchors * 4];

        // Copy m_Anchors to anchorData high speed
        Buffer.BlockCopy(m_Anchors, 0, ancohorData, 0, ancohorData.Length * sizeof(float));
        var anchors = Functional.Constant(new TensorShape(k_NumAnchors, 4), ancohorData);
        var idxScoreBox = NWSF(boxes, scores, anchors, iouThreshold, scoreThreshold);

        //input = input[.., 60..128, 60..128, ..];

        //input = Functional.Permute(input, new[] { 0, 3, 1, 2 });
        //var resized =  Functional.Interpolate(input, 
        //                                      size: new int[] {192, 192},
        //                                      mode: "linear");

        var selectedIndies = idxScoreBox.Item1;
        var selectedAnchor = Functional.IndexSelect(anchors, 0, selectedIndies);
        selectedAnchor = selectedAnchor * 128.0f;
        selectedAnchor = selectedAnchor[.., 0..2];

        var boxN = idxScoreBox.Item3;



        //var left = AnchorX - box[0, 0, 2] * 0.5f;
        //var right = AnchorX + box[0, 0, 2] * 0.5f;
        //var top = AnchorY + box[0, 0, 3] * 0.5f;
        //var bottom = AnchorY - box[0, 0, 3] * 0.5f;

        var AttentionMeshModel = graph.Compile(idxScoreBox.Item1, idxScoreBox.Item2, idxScoreBox.Item3);
        ////var AttentionMeshModel = graph.Compile(resized);
        //var AttentionMeshModel = graph.Compile(boxN,selectedAnchor);


        worker = new Worker(AttentionMeshModel, BackendType.GPUCompute);
        inputTensor = new Tensor<float>(new TensorShape(1, 128, 128, 3));
        transform = new TextureTransform()
            .SetDimensions(128, 128, 3)
            .SetTensorLayout(TensorLayout.NHWC);


        TextureConverter.ToTensor(tex, inputTensor, transform);
        worker.SetInput(0, inputTensor);

        worker.Schedule();

        //var output = worker.PeekOutput(0) as Tensor<float>;
        //if (output != null)
        //{
        //    float[] values = output.DownloadToArray();
        //    // 先頭10個だけ表示（大量の場合の例）
        //    for (int i = 0; i < Mathf.Min(10, values.Length); i++)
        //    {
        //        Debug.Log($"output[{i}] = {values[i]}");
        //    }
        //    Debug.Log($"output length: {values.Length}");
        //}
        //else
        //{
        //    Debug.LogWarning("output is null or not a Tensor<float>");
        //}

        //var output = worker.PeekOutput(0) as Tensor<float>;
        //Debug.Log("Output shape: " + output.shape);

        //renderTexture = TextureConverter.ToTexture(output);
        //rawImage.texture = renderTexture;

        var indexFace = worker.PeekOutput(0) as Tensor<int>;
        var score = worker.PeekOutput(1) as Tensor<float>;
        var Box = worker.PeekOutput(2) as Tensor<float>;

        var cpuIndex = indexFace.ReadbackAndClone();
        var cpuScore = score.ReadbackAndClone();
        var cpuBox = Box.ReadbackAndClone();

        Debug.Log(cpuIndex[0]);
        Debug.Log(cpuScore);
        Debug.Log(cpuBox[0, 0, 0] + ", " + cpuBox[0, 0, 1] + ", " + cpuBox[0, 0, 2] + ", " + cpuBox[0, 0, 3]);

        var anchorPosition = 128 * new float2(m_Anchors[cpuIndex[0], 0], m_Anchors[cpuIndex[0], 1]);
        var boxSpace = anchorPosition + new float2(cpuBox[0, 0, 0], cpuBox[0, 0, 1]);
        var boxTopRightSpace = anchorPosition + new float2(cpuBox[0, 0, 0] + 0.5f * cpuBox[0, 0, 2], cpuBox[0, 0, 1] + 0.5f * cpuBox[0, 0, 3]);

        Debug.Log("anchorPosition: " + anchorPosition);
        Debug.Log("boxSpaceUnderLeft: " + boxSpace);
        Debug.Log("boxSpaceTopRight: " + boxTopRightSpace);

    }


    void Update()
    {
        //TextureConverter.ToTensor(tex, inputTensor,transform);
        //worker.SetInput(0, inputTensor);

        //worker.Schedule();

        //var indexFace = worker.PeekOutput(0) as Tensor<int>;
        //var score = worker.PeekOutput(1) as Tensor<float>;
        //var Box = worker.PeekOutput(2) as Tensor<float>;

        //var cpuIndex = indexFace.ReadbackAndClone();
        //var cpuScore = score.ReadbackAndClone();
        //var cpuBox = Box.ReadbackAndClone();

        //Debug.Log(cpuScore);
        //Debug.Log(cpuBox);
        //Debug.Log(cpuIndex);
        

    }
}
