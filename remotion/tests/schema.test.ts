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
