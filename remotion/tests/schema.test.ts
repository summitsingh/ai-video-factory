import {readFileSync} from 'node:fs';
import {describe, expect, test} from 'vitest';
import {parseEditDocument} from '../src/schema';

const invalidCases = JSON.parse(
  readFileSync(new URL('../../tests/fixtures/edit-schema-invalid.json', import.meta.url), 'utf8'),
) as Array<{name: string; document: unknown}>;

describe('parseEditDocument', () => {
  test('rejects a scene beyond the composition', () => {
    expect(() => parseEditDocument({
      schema_version: 1, width: 1280, height: 720, fps: 30, duration_frames: 30,
      scenes: [{id: 'title', from_frame: 0, duration_frames: 31,
        title: 'Synthetic test', caption: 'Local render'}],
    })).toThrow('scene exceeds composition');
  });

  test.each(invalidCases)('rejects shared strict case: $name', ({document}) => {
    expect(() => parseEditDocument(document)).toThrow();
  });
});

describe('parseEditDocument cinematic fields', () => {
  const baseScene = () => ({
    id: 's1', from_frame: 0, duration_frames: 90,
    title: 'Synthetic test', caption: 'Local render',
  });
  const parse = (scene: unknown) => parseEditDocument({
    schema_version: 1, width: 1280, height: 720, fps: 30, duration_frames: 90,
    scenes: [scene],
  });

  test('parses act and lower_third when present', () => {
    const parsed = parse({
      ...baseScene(), act: 'ACT II', lower_third: 'Dr. Jane Park, NASA',
    });
    expect(parsed.scenes[0].act).toBe('ACT II');
    expect(parsed.scenes[0].lower_third).toBe('Dr. Jane Park, NASA');
  });

  test('defaults act and lower_third to undefined when absent', () => {
    const parsed = parse(baseScene());
    expect(parsed.scenes[0].act).toBeUndefined();
    expect(parsed.scenes[0].lower_third).toBeUndefined();
  });

  test('coerces wrong types to undefined', () => {
    const parsed = parse({...baseScene(), act: 42, lower_third: ['x']});
    expect(parsed.scenes[0].act).toBeUndefined();
    expect(parsed.scenes[0].lower_third).toBeUndefined();
  });

  test('still rejects unknown scene fields', () => {
    expect(() => parse({...baseScene(), bogus: 'x'})).toThrow('unknown field');
  });

  test('accepts a full document with both fields on multiple scenes', () => {
    const parsed = parseEditDocument({
      schema_version: 1, width: 1920, height: 1080, fps: 30, duration_frames: 180,
      scenes: [
        {...baseScene(), id: 'a', from_frame: 0, duration_frames: 90, act: 'ACT I'},
        {...baseScene(), id: 'b', from_frame: 90, duration_frames: 90, lower_third: 'Houston, TX'},
      ],
    });
    expect(parsed.scenes[0].act).toBe('ACT I');
    expect(parsed.scenes[0].lower_third).toBeUndefined();
    expect(parsed.scenes[1].act).toBeUndefined();
    expect(parsed.scenes[1].lower_third).toBe('Houston, TX');
  });
});
