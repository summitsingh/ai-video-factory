import {
  AbsoluteFill,
  Img,
  OffthreadVideo,
  Sequence,
  interpolate,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import type {EditDocument, EditScene} from './schema';

const BACKGROUNDS = [
  'linear-gradient(135deg, #071a3d 0%, #000000 100%)',
  'linear-gradient(135deg, #3d071a 0%, #000000 100%)',
  'linear-gradient(135deg, #073d2a 0%, #000000 100%)',
];

// Per-scene color grade (#6): derive a small, deterministic variation on the
// base grade from the scene seed so adjacent shots don't look identical — real
// colorists grade each shot toward a consistent target but with per-shot
// nuance. Contrast/saturation/brightness shift within a narrow band so the
// overall look stays cohesive while no two scenes match exactly.
const perSceneGrade = (seed: number): string => {
  const contrast = 1.04 + seed * 0.06; // 1.04..1.10
  const saturate = 0.89 + seed * 0.07; // 0.89..0.96
  const brightness = 0.95 + seed * 0.05; // 0.95..1.00
  const hueRotate = -2 + (seed - 0.5) * 4; // -4..0 deg
  return `contrast(${contrast.toFixed(3)}) saturate(${saturate.toFixed(3)}) brightness(${brightness.toFixed(3)}) hue-rotate(${hueRotate.toFixed(1)}deg)`;
};

const toLocalSrc = (path: string): string => {
  if (path.startsWith('http://') || path.startsWith('https://') || path.startsWith('file://')) {
    return path;
  }
  return staticFile(path);
};

// Deterministic pseudo-random per star index so the field is stable across
// frames within a scene but varied between scenes.
const starHash = (i: number): number => {
  const x = Math.sin(i * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
};

// Deterministic per-scene seed derived from the scene id, so motion, pan
// direction, and data-viz placement stay stable within a render while varying
// between scenes (gives each scene its own "feel" instead of identical motion).
const sceneSeed = (id: string): number => {
  let h = 216613626;
  for (let i = 0; i < id.length; i++) {
    h ^= id.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 100000) / 100000; // 0..1
};

// Extract up to `limit` number+unit statistics from a string (e.g. "13.5
// billion", "6.5 meters", "1.5 million km"). Returns [value, unit] pairs with
// the raw numeric value so they can be rendered as animated bars.
const extractStats = (text: string, limit = 3): Array<[number, string]> => {
  const re = /(\d+(?:\.\d+)?)\s*(billion|million|trillion|lightyears?|kilometers?|km|meters?|seconds?|minutes?|hours?|days?|years?|°C|K)\b/gi;
  const out: Array<[number, string]> = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) && out.length < limit) {
    const value = parseFloat(m[1]);
    if (!Number.isNaN(value)) out.push([value, m[2].toLowerCase()]);
  }
  return out;
};

// Human-friendly unit label for a detected statistic.
const formatUnit = (unit: string): string => {
  const map: Record<string, string> = {
    billion: 'B', million: 'M', trillion: 'T', lightyear: 'ly',
    lightyears: 'ly', kilometer: 'km', kilometers: 'km', km: 'km',
    meter: 'm', meters: 'm', second: 's', seconds: 's', minute: 'min',
    minutes: 'min', hour: 'h', hours: 'h', day: 'd', days: 'd',
    year: 'yr', years: 'yr',
  };
  return map[unit] || unit;
};

// Animated starfield: twinkling stars with slow parallax drift. Pure CSS, no
// external assets. Adds depth behind title cards and subtle motion over media.
const StarField = ({count = 90}: {count?: number}) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  return (
    <AbsoluteFill style={{pointerEvents: 'none', overflow: 'hidden'}}>
      {Array.from({length: count}, (_, i) => {
        const baseX = starHash(i) * width;
        const baseY = starHash(i + 1000) * height;
        const depth = 0.3 + starHash(i + 2000) * 0.7; // parallax factor
        const drift = frame * (4 + depth * 8);
        const twinkle =
          0.35 + 0.65 * Math.abs(Math.sin(frame * 0.04 + i));
        const size = 1 + depth * 2;
        return (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: baseX - drift,
              top: baseY,
              width: `${size}px`,
              height: `${size}px`,
              borderRadius: '50%',
              backgroundColor: '#ffffff',
              opacity: twinkle * depth,
              transform: 'translateZ(0)',
            }}
          />
        );
      })}
    </AbsoluteFill>
  );
};

const SceneMedia = ({scene}: {scene: EditScene}) => {
  const frame = useCurrentFrame();
  if (scene.clip) {
    // Per-scene color grade (#6): deterministic per-shot nuance so adjacent
    // shots don't look identical while the overall filmic target stays cohesive.
    const seed = sceneSeed(scene.id);
    return (
      <AbsoluteFill>
        <OffthreadVideo
          loop
          muted
          src={toLocalSrc(scene.clip)}
          style={{height: '100%', objectFit: 'cover', width: '100%'}}
        />
        {/* Color grade over the footage for a consistent filmic look. */}
        <AbsoluteFill style={{filter: perSceneGrade(seed)}} />
        <AbsoluteFill style={{backgroundColor: 'rgba(0, 0, 0, 0.55)'}} />
      </AbsoluteFill>
    );
  }
  if (scene.image) {
    // Per-scene Ken Burns variation (#8): deterministic pan/zoom direction and
    // speed derived from the scene id so stills never move identically.
    const seed = sceneSeed(scene.id);
    const zoom = interpolate(frame, [0, scene.duration_frames], [1, 1 + seed * 0.12], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
    // Pan direction and magnitude vary per scene (seed drives sign/scale).
    const panX = interpolate(
      frame,
      [0, scene.duration_frames],
      [(1 - seed) * 6 - 3, seed * 6 - 3],
      {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
    );
    const panY = interpolate(
      frame,
      [0, scene.duration_frames],
      [-2, 2],
      {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
    );
    // Slow ease so motion feels deliberate rather than linear.
    const eased = Math.min(1, frame / scene.duration_frames);
    // Motion blur (#4): apply a subtle Gaussian blur while the pan/zoom is in
    // progress so moving stills feel cinematic instead of jittery/strobey. The
    // blur fades out as movement completes at the end of the scene.
    const moving = Math.sin((frame / scene.duration_frames) * Math.PI); // 0..1..0 across scene
    return (
      <AbsoluteFill>
        <Img
          src={toLocalSrc(scene.image)}
          style={{
            height: '100%',
            objectFit: 'cover',
            transform: `translate(${panX}px, ${panY}px) scale(${zoom})`,
            width: '100%',
            filter: `blur(${moving * 1.2}px)`,
          }}
        />
        {/* Color grade over the still image for a consistent filmic look. */}
        <AbsoluteFill style={{filter: perSceneGrade(seed)}} />
        <AbsoluteFill style={{backgroundColor: 'rgba(0, 0, 0, 0.55)'}} />
      </AbsoluteFill>
    );
  }
  return null;
};

// Burned-in subtitle (#1): a legible caption anchored to the lower third that
// fades in/out with the scene. Uses the explicit `subtitle` field when present,
// falling back to the scene caption so every scene has one.
const Subtitle = ({text}: {text: string}) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  const fadeIn = Math.min(15, text.length > 0 ? 15 : 0);
  const enter = interpolate(frame, [0, fadeIn], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return (
    <AbsoluteFill style={{pointerEvents: 'none'}}>
      <div
        style={{
          position: 'absolute',
          bottom: height * 0.06,
          left: width * 0.05,
          right: width * 0.05,
          textAlign: 'center',
          opacity: enter,
          transform: `translateY(${-8 * (1 - enter)}px)`,
        }}
      >
        <div
          style={{
            display: 'inline-block',
            backgroundColor: 'rgba(0, 0, 0, 0.72)',
            borderRadius: 6,
            padding: `${height * 0.01}px ${width * 0.03}px`,
            color: '#f8fafc',
            fontSize: Math.round(height * 0.03),
            fontWeight: 500,
            lineHeight: 1.35,
            fontFamily: 'Arial, sans-serif',
            textShadow: '0 1px 4px rgba(0,0,0,0.9)',
          }}
        >
          {text}
        </div>
      </div>
    </AbsoluteFill>
  );
};

// Picture-in-picture inset (#5): shows the scene's secondary asset (image when a
// clip is primary, or vice versa) as a framed corner window with a subtle drop
// shadow and entrance animation.
const PictureInPicture = ({scene}: {scene: EditScene}) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  if (!scene.clip && !scene.image) return null;
  // Prefer the clip as primary (full-bleed); show image as inset. If only an
  // image exists, show nothing extra to avoid duplicating full-bleed media.
  const insetSrc = scene.clip ? scene.image : null;
  if (!insetSrc) return null;

  const size = Math.min(width, height) * 0.32;
  const enter = interpolate(frame, [0, 30], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return (
    <AbsoluteFill style={{pointerEvents: 'none'}}>
      <div
        style={{
          position: 'absolute',
          top: height * 0.1,
          right: width * 0.06,
          width: `${size}px`,
          height: `${size}px`,
          opacity: enter,
          transform: `scale(${0.8 + 0.2 * enter})`,
        }}
      >
        <div
          style={{
            position: 'relative',
            width: '100%',
            height: '100%',
            borderRadius: 8,
            overflow: 'hidden',
            boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
            border: '2px solid rgba(255,255,255,0.15)',
          }}
        >
          <Img
            src={toLocalSrc(insetSrc)}
            style={{width: '100%', height: '100%', objectFit: 'cover'}}
          />
        </div>
      </div>
    </AbsoluteFill>
  );
};

// Animated data visualization (#7): renders detected statistics as horizontal
// bars that fill on entry. Placed upper-left so it sits opposite the lower-third.
const DataViz = ({text}: {text: string}) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  const stats = extractStats(text);
  if (stats.length === 0) return null;

  const enter = interpolate(frame, [0, 45], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const barMaxWidth = Math.min(width * 0.32, 320);
  // Normalize values so the largest fills to full width (log scale for readability).
  const maxVal = Math.max(...stats.map(([v]) => v), 1);

  return (
    <AbsoluteFill style={{pointerEvents: 'none'}}>
      <div
        style={{
          position: 'absolute',
          top: height * 0.14,
          left: width * 0.06,
          backgroundColor: 'rgba(10, 18, 40, 0.78)',
          border: '1px solid rgba(34, 211, 238, 0.4)',
          borderRadius: 8,
          padding: `${height * 0.015}px ${width * 0.025}px`,
          opacity: enter,
          transform: `translateX(${-16 * (1 - enter)}px)`,
        }}
      >
        {stats.map(([value, unit], i) => (
          <div key={i} style={{marginBottom: i < stats.length - 1 ? height * 0.012 : 0}}>
            <div
              style={{
                color: '#94a3b8',
                fontSize: Math.round(height * 0.02),
                fontFamily: 'Arial, sans-serif',
                marginBottom: 3,
              }}
            >
              {formatUnit(unit)}
            </div>
            <div
              style={{
                width: `${barMaxWidth}px`,
                height: 10,
                backgroundColor: 'rgba(255,255,255,0.12)',
                borderRadius: 5,
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  width: `${Math.min(100, (value / maxVal) * 100 * enter)}%`,
                  height: '100%',
                  backgroundColor: '#22d3ee',
                  boxShadow: '0 0 8px rgba(34,211,238,0.7)',
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};

// Chapter markers (#6): a thin progress track with labeled dots marking each
// scene boundary along the bottom edge, so long-form content feels navigable.
const ChapterMarkers = ({scenes}: {scenes: EditScene[]}) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  const totalFrames = scenes.reduce((a, s) => a + s.duration_frames, 0);
  return (
    <AbsoluteFill style={{pointerEvents: 'none'}}>
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: 4,
          display: 'flex',
          alignItems: 'stretch',
        }}
      >
        {scenes.map((scene) => {
          const frac = scene.duration_frames / totalFrames;
          // Fade each marker in as its scene begins.
          const sceneStart = scenes.slice(0, scenes.indexOf(scene)).reduce((a, s) => a + s.duration_frames, 0);
          const appear = interpolate(frame - sceneStart, [0, 12], [0, 1], {
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
          });
          return (
            <div
              key={scene.id}
              style={{
                width: `${frac * 100}%`,
                backgroundColor: `rgba(34, 211, 238, ${0.5 * appear})`,
              }}
            />
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

const SceneCard = ({
  scene,
  index,
  total,
  isLast,
  sources,
  overlay,
}: {
  scene: EditScene;
  index: number;
  total: number;
  isLast: boolean;
  sources?: string[];
  overlay?: boolean;
}) => {
  const frame = useCurrentFrame();
  const {height} = useVideoConfig();
  const fadeIn = Math.min(15, scene.duration_frames);
  const fadeOut = Math.min(8, scene.duration_frames);

  const enter = interpolate(frame, [0, fadeIn], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const exit = interpolate(
    frame,
    [scene.duration_frames - fadeOut, scene.duration_frames],
    [1, 0],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
  );
  // During a transition overlay, ramp opacity in from zero so the next scene
  // dissolves over the previous one instead of popping in.
  const overlayIn = overlay
    ? interpolate(frame, [0, fadeIn + 6], [0, 1], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      })
    : 1;

  const rise = interpolate(frame, [0, fadeIn], [36, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const captionIn = interpolate(frame, [fadeIn, fadeIn + 12], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  // Intro / outro scenes render as dedicated branded sequences.
  if (scene.kind === 'intro') {
    return (
      <IntroSequence
        title={scene.title}
        subtitle={scene.caption || ''}
      />
    );
  }
  if (scene.kind === 'outro') {
    return <OutroSequence title={scene.title} sources={sources} />;
  }

  return (
    <AbsoluteFill
      style={{
        background: BACKGROUNDS[index % BACKGROUNDS.length],
        opacity: overlay ? overlayIn : Math.min(enter, exit),
      }}
    >
      <SceneMedia scene={scene} />
      {/* Picture-in-picture inset (#5): secondary asset in a corner window */}
      {scene.pip && <PictureInPicture scene={scene} />}
      {/* Animated starfield for depth behind title cards */}
      <StarField count={80} />
      {/* Cinematic vignette to focus the eye and add a filmic look */}
      <AbsoluteFill
        style={{
          background:
            'radial-gradient(ellipse at center, transparent 45%, rgba(0,0,0,0.6) 100%)',
          pointerEvents: 'none',
        }}
      />
      {/* Lower third: animated topic bar */}
      {scene.caption && <LowerThird text={scene.caption} />}
      {/* Burned-in subtitle (#1): legible caption anchored to the lower area */}
      {scene.subtitle ? (
        <Subtitle text={scene.subtitle} />
      ) : scene.caption ? (
        <Subtitle text={scene.caption} />
      ) : null}
      {/* Motion graphics: keyword-driven overlays (timeline/data/label) */}
      <MotionGraphics text={`${scene.narration || ''} ${scene.caption || ''} ${scene.title || ''}`} />
      {/* Data visualization (#7): animated bars for detected statistics */}
      <DataViz text={`${scene.narration || ''} ${scene.caption || ''} ${scene.title || ''}`} />
      {/* Keyword lower-third (#10): proper-noun/place labels from narration */}
      <KeywordLowerThird text={`${scene.narration || ''} ${scene.caption || ''} ${scene.title || ''}`} />
      {/* Animated map graphic (#5): location markers from narration keywords */}
      <MapGraphic text={`${scene.narration || ''} ${scene.caption || ''} ${scene.title || ''}`} />
      <AbsoluteFill
        style={{
          alignItems: 'center',
          display: 'flex',
          flexDirection: 'column',
          fontFamily: 'Arial, sans-serif',
          justifyContent: 'center',
          padding: '8% 10%',
          textAlign: 'center',
          transform: `translateY(${rise}px)`,
        }}
      >
        <div
          style={{
            color: '#22d3ee',
            fontSize: Math.max(20, Math.round(height * 0.032)),
            fontWeight: 700,
            letterSpacing: '0.35em',
            marginBottom: Math.round(height * 0.03),
          }}
        >
          PART {index + 1} OF {total}
          {scene.narration ? '  ·  NARRATED' : ''}
        </div>
        <div
          style={{
            color: scene.text_color ?? '#f8fafc',
            fontSize: Math.max(44, Math.round(height * 0.095)),
            fontWeight: 800,
            letterSpacing: '-0.03em',
            lineHeight: 1.08,
            maxWidth: '85%',
            textShadow: '0 2px 12px rgba(0,0,0,0.7)',
          }}
        >
          {scene.title}
        </div>
        <div
          style={{
            backgroundColor: scene.accent_color ?? '#22d3ee',
            height: Math.max(6, Math.round(height * 0.01)),
            margin: `${Math.round(height * 0.03)}px auto 0`,
            width: '16%',
          }}
        />
        <div
          style={{
            color: '#e2e8f0',
            fontSize: Math.max(24, Math.round(height * 0.042)),
            letterSpacing: '0.02em',
            lineHeight: 1.3,
            marginTop: Math.round(height * 0.03),
            maxWidth: '80%',
            opacity: captionIn,
          }}
        >
          {scene.caption}
        </div>
        {scene.visual && (
          <div
            style={{
              color: '#64748b',
              fontSize: Math.max(15, Math.round(height * 0.024)),
              fontStyle: 'italic',
              marginTop: Math.round(height * 0.02),
              opacity: captionIn,
            }}
          >
            {scene.visual}
          </div>
        )}
      </AbsoluteFill>
      {isLast && sources && sources.length > 0 && (
        <div
          style={{
            bottom: '4%',
            color: '#475569',
            fontFamily: 'Arial, sans-serif',
            fontSize: Math.max(13, Math.round(height * 0.02)),
            left: '8%',
            opacity: captionIn,
            position: 'absolute',
            right: '8%',
            textAlign: 'center',
          }}
        >
          Sources: {sources.join('  ·  ').slice(0, 160)}
        </div>
      )}
    </AbsoluteFill>
  );
};

// Keyword lower-third (#10): detects proper nouns and place names in the
// narration/caption/title and renders them as animated labels at the bottom of
// frame, so key entities are visually identified without schema changes.
const KEYWORD_LOWER_THIRDS = [
  {pattern: /\b(JWST|James Webb Space Telescope)\b/i, label: 'JWST'},
  {pattern: /\b(Sun)\b/i, label: 'The Sun'},
  {pattern: /\b(Earth)\b/i, label: 'Earth'},
  {pattern: /\b(Lagrange L2|Lagrange point|Sun Earth L2)\b/i, label: 'Lagrange L2 Point'},
  {pattern: /\b(Vera Rubin Observatory)\b/i, label: 'Vera Rubin Observatory'},
  {pattern: /\b(Mars)\b/i, label: 'Mars'},
  {pattern: /\b(Galaxy)\b/i, label: 'Galaxy'},
  {pattern: /\b(Cosmic Web|filaments)\b/i, label: 'Cosmic Web'},
];

const KeywordLowerThird = ({text}: {text: string}) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  const match = KEYWORD_LOWER_THIRDS.find((k) => k.pattern.test(text));
  if (!match) return null;

  const enter = interpolate(frame, [0, 18], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return (
    <AbsoluteFill style={{pointerEvents: 'none'}}>
      <div
        style={{
          position: 'absolute',
          bottom: height * 0.2,
          left: width * 0.06,
          opacity: enter,
          transform: `translateX(${-12 * (1 - enter)}px)`,
        }}
      >
        <div
          style={{
            backgroundColor: 'rgba(15, 23, 42, 0.85)',
            border: '1px solid rgba(148, 163, 184, 0.4)',
            borderRadius: 6,
            padding: `${height * 0.005}px ${width * 0.02}px`,
            boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
          }}
        >
          <div
            style={{
              color: '#94a3b8',
              fontSize: Math.round(height * 0.018),
              fontWeight: 600,
              letterSpacing: '0.05em',
              textTransform: 'uppercase',
            }}
          >
            {match.label}
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};

// Animated map graphic (#5): renders a simple stylized location marker when
// place keywords are detected in the narration, so viewers can follow where
// things happen. No schema changes required — keyword-driven only.
const MAP_KEYWORDS = [
  {pattern: /\b(Sun)\b/i, x: 0.2, y: 0.3},
  {pattern: /\b(Earth)\b/i, x: 0.35, y: 0.45},
  {pattern: /\b(Mars)\b/i, x: 0.5, y: 0.4},
  {pattern: /\b(Lagrange L2|Lagrange point|Sun Earth L2)\b/i, x: 0.65, y: 0.35},
  {pattern: /\b(Vera Rubin Observatory|Chile)\b/i, x: 0.25, y: 0.7},
];

const MapGraphic = ({text}: {text: string}) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  const match = MAP_KEYWORDS.find((k) => k.pattern.test(text));
  if (!match) return null;

  const enter = interpolate(frame, [0, 30], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const dotX = match.x * width;
  const dotY = match.y * height;
  return (
    <AbsoluteFill style={{pointerEvents: 'none'}}>
      {/* Simple map background */}
      <div
        style={{
          position: 'absolute',
          top: height * 0.1,
          left: width * 0.1,
          width: `${width * 0.8}px`,
          height: `${height * 0.8}px`,
          backgroundColor: 'rgba(15, 23, 42, 0.6)',
          border: '1px solid rgba(148, 163, 184, 0.3)',
          borderRadius: 8,
          opacity: enter * 0.9,
        }}
      />
      {/* Animated location dot */}
      <div
        style={{
          position: 'absolute',
          left: `${dotX}px`,
          top: `${dotY}px`,
          width: 12,
          height: 12,
          borderRadius: '50%',
          backgroundColor: '#22d3ee',
          boxShadow: '0 0 16px rgba(34,211,238,0.8)',
          opacity: enter,
          transform: `scale(${enter})`,
        }}
      />
    </AbsoluteFill>
  );
};

const ProgressBar = () => {
  const frame = useCurrentFrame();
  const {durationInFrames} = useVideoConfig();
  const progress = interpolate(frame, [0, durationInFrames], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return (
    <div
      style={{
        backgroundColor: 'rgba(148, 163, 184, 0.25)',
        bottom: 0,
        height: 8,
        left: 0,
        position: 'absolute',
        right: 0,
      }}
    >
      <div
        style={{
          backgroundColor: '#22d3ee',
          height: '100%',
          width: `${progress * 100}%`,
        }}
      />
    </div>
  );
};

// Motion graphics: keyword-driven overlays that animate in when relevant terms
// appear in the narration/caption/title. Detects dates, measurements, and
// distances to render timelines, data callouts, and labels — no schema changes
// or worker re-runs required.
const MOTION_KEYWORDS = [
  {pattern: /launched|december 2021|ariane/i, type: 'timeline'},
  {pattern: /6\.5 meters|hexagonal|gold mirror/i, type: 'data'},
  {pattern: /billion lightyears|redshift|320 million|13\.\d billion/i, type: 'data'},
  {pattern: /infrared|invisible|dust cloud/i, type: 'label'},
  {pattern: /lagrange|l2 point|1\.5 million km|orbit/i, type: 'label'},
];

const MotionGraphics = ({text}: {text: string}) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  const match = MOTION_KEYWORDS.find((k) => k.pattern.test(text));
  if (!match) return null;

  // Animate in over first 40 frames.
  const enter = interpolate(frame, [0, 40], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  if (match.type === 'timeline') {
    return (
      <AbsoluteFill style={{pointerEvents: 'none'}}>
        {/* Animated timeline bar */}
        <div
          style={{
            position: 'absolute',
            top: height * 0.2,
            left: width * 0.1,
            width: `${width * 0.8}px`,
            height: 4,
            backgroundColor: 'rgba(34, 211, 238, 0.3)',
            borderRadius: 2,
          }}
        >
          <div
            style={{
              height: '100%',
              width: `${enter * 100}%`,
              backgroundColor: '#22d3ee',
              boxShadow: '0 0 12px rgba(34,211,238,0.8)',
            }}
          />
        </div>
      </AbsoluteFill>
    );
  }

  if (match.type === 'data') {
    return (
      <AbsoluteFill style={{pointerEvents: 'none'}}>
        {/* Data callout box */}
        <div
          style={{
            position: 'absolute',
            top: height * 0.15,
            right: width * 0.08,
            backgroundColor: 'rgba(34, 211, 238, 0.15)',
            border: '1px solid rgba(34, 211, 238, 0.6)',
            borderRadius: 8,
            padding: `${height * 0.02}px ${width * 0.03}px`,
            opacity: enter,
            transform: `translateX(${20 * (1 - enter)}px)`,
          }}
        >
          <div
            style={{
              color: '#22d3ee',
              fontSize: Math.round(height * 0.03),
              fontWeight: 700,
              letterSpacing: '0.05em',
            }}
          >
            ▸ COSMIC DATA
          </div>
        </div>
      </AbsoluteFill>
    );
  }

  // label type
  return (
    <AbsoluteFill style={{pointerEvents: 'none'}}>
      {/* Animated label tag */}
      <div
        style={{
          position: 'absolute',
          top: height * 0.18,
          left: width * 0.08,
          backgroundColor: 'rgba(168, 85, 247, 0.2)',
          border: '1px solid rgba(168, 85, 247, 0.6)',
          borderRadius: 6,
          padding: `${height * 0.01}px ${width * 0.03}px`,
          opacity: enter,
          transform: `translateX(${-20 * (1 - enter)}px)`,
        }}
      >
        <div
          style={{
            color: '#c084fc',
            fontSize: Math.round(height * 0.025),
            fontWeight: 600,
            letterSpacing: '0.08em',
          }}
        >
          ◈ KEY CONCEPT
        </div>
      </div>
    </AbsoluteFill>
  );
};

const TRANSITION_FRAMES = 24; // ~0.8s cross-dissolve at 30fps

// Intro sequence: branded title card with animated reveal.
const IntroSequence = ({title, subtitle}: {title: string; subtitle: string}) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  const titleIn = interpolate(frame, [0, 30], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const subtitleIn = interpolate(frame, [24, 54], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const lineExpand = interpolate(frame, [18, 48], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return (
    <AbsoluteFill>
      <StarField count={120} />
      <AbsoluteFill
        style={{
          background:
            'radial-gradient(ellipse at center, #0a1a3d 0%, #000000 100%)',
        }}
      />
      <AbsoluteFill
        style={{
          alignItems: 'center',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          fontFamily: 'Arial, sans-serif',
          textAlign: 'center',
        }}
      >
        <div
          style={{
            color: '#22d3ee',
            fontSize: Math.round(height * 0.03),
            fontWeight: 700,
            letterSpacing: '0.4em',
            opacity: titleIn,
            transform: `translateY(${-10 * (1 - titleIn)}px)`,
          }}
        >
          A DOCUMENTARY PRESENTATION
        </div>
        <div
          style={{
            color: '#f8fafc',
            fontSize: Math.round(height * 0.075),
            fontWeight: 800,
            letterSpacing: '-0.03em',
            lineHeight: 1.1,
            marginTop: height * 0.04,
            maxWidth: '90%',
            textShadow: '0 4px 24px rgba(0,0,0,0.8)',
            opacity: titleIn,
            transform: `translateY(${-20 * (1 - titleIn)}px)`,
          }}
        >
          {title}
        </div>
        <div
          style={{
            height: 4,
            width: `${lineExpand * 200}px`,
            backgroundColor: '#22d3ee',
            marginTop: height * 0.04,
            opacity: titleIn,
            boxShadow: '0 0 16px rgba(34,211,238,0.6)',
          }}
        />
        <div
          style={{
            color: '#cbd5e1',
            fontSize: Math.round(height * 0.035),
            fontWeight: 500,
            letterSpacing: '0.05em',
            marginTop: height * 0.04,
            opacity: subtitleIn,
            transform: `translateY(${-15 * (1 - subtitleIn)}px)`,
          }}
        >
          {subtitle}
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

// Outro/credits sequence.
const OutroSequence = ({title, sources}: {title: string; sources?: string[]}) => {
  const frame = useCurrentFrame();
  const {height} = useVideoConfig();
  const contentIn = interpolate(frame, [0, 42], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return (
    <AbsoluteFill>
      <StarField count={100} />
      <AbsoluteFill
        style={{
          background:
            'radial-gradient(ellipse at center, #1a0a2d 0%, #000000 100%)',
        }}
      />
      <AbsoluteFill
        style={{
          alignItems: 'center',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          fontFamily: 'Arial, sans-serif',
          textAlign: 'center',
          opacity: contentIn,
          transform: `translateY(${-15 * (1 - contentIn)}px)`,
        }}
      >
        <div
          style={{
            color: '#f8fafc',
            fontSize: Math.round(height * 0.06),
            fontWeight: 800,
            letterSpacing: '-0.02em',
            lineHeight: 1.15,
            maxWidth: '90%',
            textShadow: '0 4px 24px rgba(0,0,0,0.8)',
          }}
        >
          {title}
        </div>
        <div
          style={{
            height: 3,
            width: 160,
            backgroundColor: '#a855f7',
            marginTop: height * 0.03,
            marginBottom: height * 0.04,
          }}
        />
        <div
          style={{
            color: '#94a3b8',
            fontSize: Math.round(height * 0.028),
            fontWeight: 500,
            letterSpacing: '0.1em',
          }}
        >
          PRODUCTION · AI VIDEO FACTORY
        </div>
        {sources && sources.length > 0 && (
          <div
            style={{
              color: '#64748b',
              fontSize: Math.round(height * 0.02),
              marginTop: height * 0.03,
              maxWidth: '85%',
              fontStyle: 'italic',
            }}
          >
            Sources: {sources.join(' · ').slice(0, 140)}
          </div>
        )}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

// Lower third: animated topic bar that slides in at the bottom of frame.
const LowerThird = ({text}: {text: string}) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  const enter = interpolate(frame, [0, 18], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return (
    <AbsoluteFill
      style={{
        bottom: height * 0.14,
        left: width * 0.06,
        opacity: enter,
        transform: `translateX(${-20 * (1 - enter)}px)`,
      }}
    >
      <div
        style={{
          backgroundColor: 'rgba(34, 211, 238, 0.9)',
          padding: `${height * 0.01}px ${width * 0.03}px`,
          borderRadius: 6,
          boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
        }}
      >
        <div
          style={{
            color: '#0a0a0a',
            fontSize: Math.round(height * 0.03),
            fontWeight: 700,
            letterSpacing: '0.02em',
          }}
        >
          {text}
        </div>
      </div>
    </AbsoluteFill>
  );
};

export const SyntheticVideo = ({scenes, sources}: EditDocument) => {
  return (
    <AbsoluteFill>
      {scenes.map((scene, index) => (
        <Sequence
          durationInFrames={scene.duration_frames}
          from={scene.from_frame}
          key={scene.id}
          layout="none"
        >
          <SceneCard
            index={index}
            isLast={index === scenes.length - 1}
            scene={scene}
            sources={sources}
            total={scenes.length}
          />
        </Sequence>
      ))}
      {/* Chapter markers (#6): full-composition progress track with per-scene
          segments so long-form content feels navigable. */}
      <Sequence durationInFrames={scenes.reduce((a, s) => a + s.duration_frames, 0)}>
        <ChapterMarkers scenes={scenes} />
      </Sequence>
      {/* Cross-dissolve transitions: fade the next scene in over the tail of
          the current one. Only valid within a chunk (boundaries hard-cut). */}
      {scenes.slice(0, -1).map((scene, index) => (
        <Sequence
          durationInFrames={TRANSITION_FRAMES}
          from={scene.from_frame + scene.duration_frames - TRANSITION_FRAMES}
          key={`transition-${scene.id}`}
          layout="none"
        >
          <SceneCard
            index={index + 1}
            isLast={index === scenes.length - 2}
            scene={scenes[index + 1]}
            sources={sources}
            total={scenes.length}
            overlay
          />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};
