Shader "Unlit/IrisDraw"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _LeftIrisX("Left Iris X", Float) = 0
        _LeftIrisY("Left Iris Y", Float) = 0
        _RightIrisX("Right Iris X", Float) = 0
        _RightIrisY("Right Iris Y", Float) = 0
        _DotRadius("Dot Radius", Float) = 0.01
        _DotColor("Dot Color", Color) = (1, 0, 0, 1)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float _LeftIrisX, _LeftIrisY;
            float _RightIrisX, _RightIrisY;
            float _DotRadius;
            fixed4 _DotColor;

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float2 uv = i.uv;
                uv = float2(uv.x, 1.0 - uv.y); // Y軸を反転
                float2 left = float2(_LeftIrisX, _LeftIrisY)/ 192;
                float2 right = float2(_RightIrisX, _RightIrisY) /192;

                // ベースカラー
                fixed4 baseColor = tex2D(_MainTex, i.uv);

                if (distance(uv, left) < _DotRadius || distance(uv, right) < _DotRadius)
                    return _DotColor;

                return baseColor;
            }
            ENDCG
        }
    }
}
